"""
Frame Sequence Generator — extracts frame windows from video,
saves them to ComfyUI output, and exposes both IMAGE tensors
and HTTP paths for downstream nodes.
"""
import os
import shutil
import subprocess
from pathlib import Path

import folder_paths
import numpy as np
import torch
from PIL import Image


class FrameSequenceGenerator:
    """Extract frame windows from video or provided IMAGE batch around detected beats.

    Works in DUO with BeatIt: receives beats_json, extracts high-FPS
    frame windows around each beat time.

    Two modes:
      IMAGE mode: connect images (e.g. Load Image Batch) — slices the
                  batch by beat windows. No ffmpeg, no video file needed.
      VIDEO mode: provide video path — extracts frames via ffmpeg.

    Without beats: uses start_seconds/end_seconds as a manual range.
    When audio (AUDIO) is connected and no beats are provided,
    end_seconds is auto-set to the audio duration.

    Output:
      - images: IMAGE batch [N, H, W, C]
      - http_paths: newline-separated HTTP URLs (empty in IMAGE mode)
      - local_paths: newline-separated absolute paths (empty in IMAGE mode)
      - frame_count: int
      - beats_used: JSON of which beats were processed
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("STRING", {"default": "", "multiline": False, "placeholder": "Leave empty when images input is connected"}),
                "base_url": ("STRING", {"default": "http://localhost:8188", "multiline": False}),
                "filename_prefix": ("STRING", {"default": "frame_seq", "multiline": False}),
                "fps": ("FLOAT", {"default": 30.0, "min": 1.0, "max": 120.0, "step": 1.0}),
                "window_seconds": ("FLOAT", {"default": 1.0, "min": 0.2, "max": 5.0, "step": 0.1, "tooltip": "Seconds before+after each beat"}),
            },
            "optional": {
                "images": ("IMAGE",),
                "beats_json": ("STRING", {"default": "", "multiline": True, "placeholder": "JSON from BeatIt node (optional, overrides manual range)"}),
                "drops_only": ("BOOLEAN", {"default": False, "tooltip": "Only extract frames around beats marked as drops (is_drop=true)"}),
                "main_job_only": ("BOOLEAN", {"default": False, "tooltip": "Only extract frames around the strongest drop (highest energy_jump)"}),
                "beat_selection_mode": (["legacy", "all_beats", "drops_only", "beats_before_drop", "beats_after_drop"], {
                    "default": "legacy",
                    "tooltip": "Select all beats, drops, or beats strictly before/after an anchor drop. legacy preserves the two boolean filters.",
                }),
                "anchor_drop_strategy": (["first_downbeat_drop", "first_drop", "strongest_drop"], {
                    "default": "first_downbeat_drop",
                    "tooltip": "How before/after modes choose their anchor drop.",
                }),
                "ignore_start_seconds": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 30.0, "step": 0.1,
                    "tooltip": "Ignore startup transients before this time for explicit beat-selection modes.",
                }),
                "max_beat_time_seconds": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 86400.0, "step": 0.001,
                    "tooltip": "Optional upper beat-time bound; 0 disables it. Set to the video duration to reject boundary predictions.",
                }),
                "start_seconds": ("FLOAT", {"default": 0.0, "min": 0.0, "step": 0.1}),
                "end_seconds": ("FLOAT", {"default": 2.0, "min": 0.1, "step": 0.1}),
                "audio": ("AUDIO",),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "INT", "STRING")
    RETURN_NAMES = ("images", "http_paths", "local_paths", "frame_count", "beats_used")
    FUNCTION = "generate"
    CATEGORY = "Amin/Beatdrop"

    @staticmethod
    def _frame_timestamps(source_indices, batch_offset, fps, start_time_seconds=None):
        """Create stable source↔batch frame mappings with exact timestamps."""
        fps = max(1e-6, float(fps))
        return [
            {
                "batch_index": int(batch_offset + local_index),
                "source_frame_index": int(source_index),
                "time_seconds": round(
                    float(start_time_seconds) + local_index / fps
                    if start_time_seconds is not None
                    else float(source_index) / fps,
                    6,
                ),
            }
            for local_index, source_index in enumerate(source_indices)
        ]

    @staticmethod
    def _select_beats_for_mode(beats, mode="legacy", anchor_strategy="first_downbeat_drop",
                               ignore_start_seconds=1.0, max_time_seconds=0.0):
        """Select beat phases around one stable anchor drop without mutating input."""
        mode = str(mode or "legacy").strip()
        ordered = sorted(
            (dict(beat) for beat in beats if isinstance(beat, dict)),
            key=lambda beat: float(beat.get("time_seconds", 0.0)),
        )
        if mode == "legacy":
            return ordered, None

        start = max(0.0, float(ignore_start_seconds))
        maximum = float(max_time_seconds or 0.0)
        eligible = [
            beat for beat in ordered
            if float(beat.get("time_seconds", 0.0)) >= start
            and (maximum <= 0.0 or float(beat.get("time_seconds", 0.0)) <= maximum)
        ]
        if mode == "all_beats":
            return eligible, None
        if mode == "drops_only":
            return [beat for beat in eligible if beat.get("is_drop", False)], None
        if mode not in {"beats_before_drop", "beats_after_drop"}:
            raise ValueError(f"Unknown beat_selection_mode: {mode}")

        drops = [beat for beat in eligible if beat.get("is_drop", False)]
        if not drops:
            raise RuntimeError(f"{mode} requires at least one drop after {start:.3f}s")

        strategy = str(anchor_strategy or "first_downbeat_drop").strip()
        if strategy == "first_downbeat_drop":
            anchor = next((beat for beat in drops if beat.get("is_downbeat", False)), drops[0])
        elif strategy == "first_drop":
            anchor = drops[0]
        elif strategy == "strongest_drop":
            anchor = max(
                drops,
                key=lambda beat: (
                    float(beat.get("energy_jump", 0.0)),
                    float(beat.get("drop_confidence", 0.0)),
                ),
            )
        else:
            raise ValueError(f"Unknown anchor_drop_strategy: {strategy}")

        anchor_time = float(anchor.get("time_seconds", 0.0))
        if mode == "beats_before_drop":
            selected = [beat for beat in eligible if float(beat.get("time_seconds", 0.0)) < anchor_time]
            relation = "before"
        else:
            selected = [beat for beat in eligible if float(beat.get("time_seconds", 0.0)) > anchor_time]
            relation = "after"

        annotated = [
            {
                **beat,
                "selection_mode": mode,
                "relative_to_anchor": relation,
                "anchor_drop_time_seconds": round(anchor_time, 6),
                "anchor_drop_frame_index": int(anchor.get("frame_index", 0)),
                "anchor_drop_strategy": strategy,
            }
            for beat in selected
        ]
        return annotated, dict(anchor)

    def generate(self, video, base_url, filename_prefix, fps, window_seconds,
                 images=None, beats_json="", drops_only=False, main_job_only=False,
                 beat_selection_mode="legacy", anchor_drop_strategy="first_downbeat_drop",
                 ignore_start_seconds=1.0, max_beat_time_seconds=0.0,
                 start_seconds=0.0, end_seconds=2.0, audio=None):
        fps = max(1.0, min(float(fps), 120.0))
        window = max(0.2, float(window_seconds))

        # Parse beats from BeatIt if provided
        beats = []
        if beats_json and beats_json.strip():
            try:
                beats = _json.loads(beats_json)
            except Exception:
                pass
        if not isinstance(beats, list):
            beats = []

        if str(beat_selection_mode).strip() != "legacy" and beats:
            beats, _anchor = self._select_beats_for_mode(
                beats,
                mode=beat_selection_mode,
                anchor_strategy=anchor_drop_strategy,
                ignore_start_seconds=ignore_start_seconds,
                max_time_seconds=max_beat_time_seconds,
            )
            if not beats:
                raise RuntimeError(f"beat_selection_mode={beat_selection_mode} selected no beats")
        else:
            # Backward-compatible boolean filters.
            if drops_only and beats:
                beats = [b for b in beats if b.get("is_drop", False)]
                if not beats:
                    raise RuntimeError("drops_only=True but no beats marked as drops in beats_json")

            if main_job_only and beats:
                def _drop_strength(b):
                    return float(b.get("energy_jump", 0)) or float(b.get("drop_confidence", 0))
                beats = [max(beats, key=_drop_strength)]

        # --- IMAGE mode: slice existing batch by beat windows ---
        if images is not None:
            if not isinstance(images, torch.Tensor):
                raise ValueError("images must be an IMAGE tensor")
            B = images.shape[0]

            # Build frame index ranges from beats
            if beats:
                idx_ranges = []
                beats_used = []
                for bi, b in enumerate(beats):
                    t = float(b.get("time_seconds", 0))
                    f0 = max(0, int((t - window) * fps))
                    f1 = min(B, int((t + window) * fps) + 1)
                    if f1 > f0:
                        idx_ranges.append((f0, f1))
                        beats_used.append({"beat_index": bi, **b,
                                           "range_start": round(t - window, 3),
                                           "range_end": round(t + window, 3)})
            else:
                # Manual range: convert seconds to frame indices
                f0 = max(0, int(float(start_seconds) * fps))
                actual_end = float(end_seconds)
                if audio is not None and isinstance(audio, dict) and "waveform" in audio:
                    waveform = audio["waveform"]
                    sr = audio["sample_rate"]
                    if waveform.dim() == 3:
                        waveform = waveform[0]
                    ns = waveform.shape[-1] if waveform.dim() == 2 else waveform.shape[0]
                    dur = ns / max(1, int(sr))
                    if dur > actual_end:
                        actual_end = dur
                f1 = min(B, int(actual_end * fps) + 1)
                idx_ranges = [(f0, f1)]
                beats_used = [{
                    "beat_index": None,
                    "time_seconds": round(float(start_seconds), 3),
                    "frame_index": f0,
                    "is_drop": False,
                    "fallback": "manual_range",
                    "range_start": round(float(start_seconds), 3),
                    "range_end": round(float(actual_end), 3),
                }]

            # Collect frames from all ranges, tracking batch offsets
            selected = []
            batch_offset = 0
            for wi, (f0, f1) in enumerate(idx_ranges):
                chunk = images[f0:f1]
                selected.append(chunk)
                n_frames = chunk.shape[0]
                if wi < len(beats_used):
                    beats_used[wi]["batch_offset"] = batch_offset
                    beats_used[wi]["batch_frame_count"] = n_frames
                    beats_used[wi]["frames"] = self._frame_timestamps(
                        range(f0, f1), batch_offset, fps,
                    )
                batch_offset += n_frames
            if not selected:
                raise RuntimeError("No frames in selected ranges")
            batch = torch.cat(selected, dim=0)

            return (
                batch,
                "",   # http_paths — not available in image mode
                "",   # local_paths — not available in image mode
                batch.shape[0],
                _json.dumps(beats_used),
            )

        # --- VIDEO mode: extract frames via ffmpeg ---
        if not video or not os.path.isfile(video):
            raise ValueError(f"Video file not found: {video}")

        # Build time ranges: one per beat, or single manual range
        ranges = []
        if beats:
            for b in beats:
                t = float(b.get("time_seconds", 0))
                ranges.append((max(0, t - window), t + window))
        else:
            # Auto end_seconds from audio duration if audio is connected
            actual_end = float(end_seconds)
            if audio is not None and isinstance(audio, dict) and "waveform" in audio:
                waveform = audio["waveform"]
                sample_rate = audio["sample_rate"]
                if waveform.dim() == 3:
                    waveform = waveform[0]
                if waveform.dim() == 2:
                    num_samples = waveform.shape[-1]
                else:
                    num_samples = waveform.shape[0]
                audio_duration = num_samples / max(1, int(sample_rate))
                # Only override if user didn't explicitly set a longer end_seconds
                if audio_duration > float(end_seconds):
                    actual_end = audio_duration
            ranges.append((float(start_seconds), actual_end))

        out_dir = Path(folder_paths.get_output_directory()) / "frame_sequences"
        base_url = base_url.rstrip("/")
        all_images = []
        all_http = []
        all_local = []
        beats_used = []
        batch_offset = 0

        for ri, (t_start, t_end) in enumerate(ranges):
            duration = max(0.1, t_end - t_start)
            run_id = f"{filename_prefix}_r{ri}_{int(t_start * 1000)}"
            frame_dir = out_dir / run_id
            frame_dir.mkdir(parents=True, exist_ok=True)

            cmd = [
                "ffmpeg", "-hide_banner", "-loglevel", "error",
                "-ss", str(t_start), "-t", str(duration),
                "-i", str(video),
                "-vf", f"fps={fps}",
                "-q:v", "2",
                "-frames:v", str(int(duration * fps) + 1),
                str(frame_dir / "frame_%06d.jpg"),
            ]
            subprocess.run(cmd, check=True, capture_output=True, timeout=300)

            frame_files = sorted(frame_dir.glob("frame_*.jpg"))
            n_frames = len(frame_files)
            for fpath in frame_files:
                img = Image.open(fpath).convert("RGB")
                arr = np.array(img, dtype=np.float32) / 255.0
                all_images.append(torch.from_numpy(arr))
                all_local.append(str(fpath))
                rel = fpath.relative_to(folder_paths.get_output_directory())
                all_http.append(f"{base_url}/view?filename={rel.name}&subfolder={rel.parent}&type=output")

            if ri < len(beats):
                beats_used.append({"beat_index": ri, **beats[ri],
                                   "range_start": round(t_start, 3),
                                   "range_end": round(t_end, 3),
                                   "batch_offset": batch_offset,
                                   "batch_frame_count": n_frames,
                                   "frames": self._frame_timestamps(
                                       range(int(round(t_start * fps)), int(round(t_start * fps)) + n_frames),
                                       batch_offset, fps, start_time_seconds=t_start,
                                   )})
            else:
                source_start = int(round(t_start * fps))
                beats_used.append({
                    "beat_index": None,
                    "time_seconds": round(t_start, 3),
                    "frame_index": source_start,
                    "is_drop": False,
                    "fallback": "manual_range",
                    "range_start": round(t_start, 3),
                    "range_end": round(t_end, 3),
                    "batch_offset": batch_offset,
                    "batch_frame_count": n_frames,
                    "frames": self._frame_timestamps(
                        range(source_start, source_start + n_frames),
                        batch_offset, fps, start_time_seconds=t_start,
                    ),
                })
            batch_offset += n_frames

        if not all_images:
            raise RuntimeError("No frames extracted from video")

        batch = torch.stack(all_images, dim=0)
        return (
            batch,
            "\n".join(all_http),
            "\n".join(all_local),
            len(all_images),
            _json.dumps(beats_used),
        )


import json as _json
import wave as _wave
import math as _math
import urllib.error as _urlerror
import urllib.request as _urlrequest


def _extract_first_json_object(text):
    decoder = _json.JSONDecoder()
    source = str(text or "")
    for idx, ch in enumerate(source):
        if ch not in "[{":
            continue
        try:
            value, _end = decoder.raw_decode(source[idx:])
            return value
        except Exception:
            continue
    return None


def _extract_chat_content(response_json):
    if not isinstance(response_json, dict):
        return str(response_json)
    choices = response_json.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0] if isinstance(choices[0], dict) else {}
        message_candidate = first.get("message")
        message = message_candidate if isinstance(message_candidate, dict) else {}
        content = message.get("content")
        if content is not None:
            return content if isinstance(content, str) else _json.dumps(content, ensure_ascii=False)
        text = first.get("text")
        if text is not None:
            return str(text)
    for key in ("content", "response", "text", "message"):
        value = response_json.get(key)
        if isinstance(value, str):
            return value
    return _json.dumps(response_json, ensure_ascii=False)


def _post_json(url, headers, payload, timeout):
    data = _json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = _urlrequest.Request(url, data=data, headers=headers, method="POST")
    try:
        with _urlrequest.urlopen(req, timeout=max(5, int(timeout))) as response:
            raw = response.read().decode("utf-8", errors="replace")
            try:
                return raw, _json.loads(raw)
            except Exception:
                return raw, {"raw": raw}
    except _urlerror.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"AlphaRavis HTTP {exc.code}: {body[:1000]}") from exc


class BeatItNode:
    """Detect beat drops from audio or video.

    Uses beat_this (https://github.com/CPJKU/beat_this) for high-quality
    beat/downbeat detection. Falls back to RMS energy jump detection
    if beat_this is unavailable or fails.

    Input priority: AUDIO tensor > video > audio_path

    Inputs:
      - fps: video FPS for frame index mapping
      - audio: ComfyUI AUDIO tensor (drag any audio node output here)
      - audio_path: path to wav/mp3 file (optional)
      - video: video path — audio is extracted via ffmpeg (optional)

    Outputs:
      - beats_json: JSON array of {time_seconds, frame_index, confidence}
      - beat_count: number of detected beats
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "fps": ("FLOAT", {"default": 30.0, "min": 1.0, "max": 120.0, "step": 1.0}),
            },
            "optional": {
                "audio": ("AUDIO",),
                "audio_path": ("STRING", {"default": "", "multiline": False}),
                "video": ("STRING", {"default": "", "multiline": False, "placeholder": "Video path — audio is auto-extracted via ffmpeg"}),
                "max_beats": ("INT", {"default": 64, "min": 1, "max": 512, "step": 1,
                                      "tooltip": "Maximum beat events returned; raise for long or fast clips."}),
            },
        }

    RETURN_TYPES = ("STRING", "INT")
    RETURN_NAMES = ("beats_json", "beat_count")
    FUNCTION = "detect"
    CATEGORY = "Amin/Beatdrop"

    def _detect_with_beatthis(self, audio_path, fps):
        import numpy as np
        from beat_this.inference import File2Beats
        from beat_this.preprocessing import load_audio

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        float16 = device.startswith("cuda")
        predictor = getattr(self, "_beatthis_predictor", None)
        predictor_key = (device, float16)
        if predictor is None or getattr(self, "_beatthis_predictor_key", None) != predictor_key:
            predictor = File2Beats(
                checkpoint_path="final0",
                device=device,
                float16=float16,
                dbn=False,
            )
            self._beatthis_predictor = predictor
            self._beatthis_predictor_key = predictor_key

        beats, downbeats = predictor(audio_path)
        audio, sr = load_audio(audio_path)

        if not len(beats):
            return []

        # Compute RMS energy in short windows to detect energy jumps (drops)
        hop = int(sr * 0.01)
        n_samples = len(audio)
        if hasattr(audio, 'numpy'):
            audio_np = audio.numpy()
        else:
            audio_np = np.array(audio)

        rms_times = []
        rms_values = []
        for start in range(0, n_samples - hop, hop):
            chunk = audio_np[start:start + hop]
            rms = float(np.sqrt(np.mean(chunk ** 2)))
            rms_times.append(start / sr)
            rms_values.append(rms)

        rms_values = np.array(rms_values)
        if len(rms_values) < 4:
            rms_values = np.ones(4) * 0.01

        median_rms = float(np.median(rms_values))
        max_rms = float(np.max(rms_values)) or 1e-9

        result = []
        for i, t in enumerate(beats):
            t = float(t)
            frame = int(round(t * fps))

            # Energy before and after this beat (200ms windows)
            before_start = max(0, int((t - 0.2) * sr / hop))
            before_end = max(0, int(t * sr / hop))
            after_start = min(len(rms_values) - 1, int(t * sr / hop))
            after_end = min(len(rms_values) - 1, int((t + 0.3) * sr / hop))

            energy_before = float(np.mean(rms_values[before_start:before_end])) if before_end > before_start else 0.0
            energy_after = float(np.mean(rms_values[after_start:after_end])) if after_end > after_start else 0.0

            # Drop score: how much energy jumps up
            if energy_before > 1e-9:
                jump = energy_after / energy_before
            else:
                jump = 1.0

            is_downbeat = any(abs(t - float(db)) < 0.05 for db in downbeats)
            is_drop = jump >= 1.8 or (jump >= 1.4 and is_downbeat)
            drop_confidence = max(0.0, min(1.0, (jump - 1.0) / 1.5))

            result.append({
                "time_seconds": round(t, 3),
                "frame_index": frame,
                "confidence": round(min(1.0, 0.5 + drop_confidence * 0.5), 3),
                "is_drop": is_drop,
                "drop_confidence": round(drop_confidence, 3),
                "is_downbeat": is_downbeat,
                "energy_jump": round(jump, 2),
                "method": "beat_this",
            })
        return result

    def _detect_with_rms(self, audio_path, fps, max_beats=64):
        if not audio_path or not os.path.isfile(audio_path):
            raise ValueError(f"Audio file not found: {audio_path}")

        fps = max(1.0, float(fps))
        try:
            import wave
            with wave.open(audio_path, "rb") as wf:
                nchannels = wf.getnchannels()
                sampwidth = wf.getsampwidth()
                framerate = wf.getframerate()
                nframes = wf.getnframes()
                raw = wf.readframes(nframes)
        except Exception:
            # Try ffmpeg convert to wav
            tmp_wav = audio_path + ".tmp_beatit.wav"
            subprocess.run([
                "ffmpeg", "-hide_banner", "-loglevel", "error",
                "-i", audio_path, "-ac", "1", "-ar", "22050",
                "-sample_fmt", "s16", "-y", tmp_wav,
            ], check=True, timeout=120, capture_output=True)
            with _wave.open(tmp_wav, "rb") as wf:
                nchannels = wf.getnchannels()
                sampwidth = wf.getsampwidth()
                framerate = wf.getframerate()
                nframes = wf.getnframes()
                raw = wf.readframes(nframes)
            try:
                os.unlink(tmp_wav)
            except Exception:
                pass

        divisor = 2 ** (8 * sampwidth) // 2
        if sampwidth == 1:
            import struct
            samples = [struct.unpack_from("b", raw, i)[0] / 127.0 for i in range(nframes)]
        elif sampwidth == 2:
            import struct
            samples = [struct.unpack_from("<h", raw, i)[0] / 32768.0 for i in range(0, len(raw), 2)]
        else:
            samples = [0.0]

        # RMS envelope
        window_ms = 80
        hop_ms = 20
        window = max(1, int(framerate * window_ms / 1000))
        hop = max(1, int(framerate * hop_ms / 1000))
        envelope = []
        for start in range(0, len(samples) - window, hop):
            chunk = samples[start: start + window]
            if not chunk:
                continue
            rms = _math.sqrt(sum(v * v for v in chunk) / len(chunk))
            center = (start + len(chunk) / 2) / framerate
            envelope.append((center, rms))

        if len(envelope) < 4:
            return ("[]", 0)

        times = [e[0] for e in envelope]
        rms_vals = [e[1] for e in envelope]
        max_rms = max(rms_vals) or 1e-9
        beats = []

        def _local_median(values, a, b):
            subset = values[max(0, a): min(len(values), b)]
            if not subset:
                return 0.001
            return float(sorted(subset)[len(subset) // 2])

        for i in range(2, len(envelope)):
            current = rms_vals[i]
            prev = _local_median(rms_vals, i - 12, i - 1)
            after = _local_median(rms_vals, i, min(len(rms_vals), i + 6))
            jump = max(current, after) / prev if prev > 1e-9 else 1.0
            norm_energy = current / max_rms
            confidence = min(1.0, (jump - 1.0) / 3.0 * 0.7 + norm_energy * 0.3)
            if jump >= 1.65 and norm_energy >= 0.12 and confidence >= 0.3:
                time_sec = times[i]
                frame_idx = int(round(time_sec * fps))
                is_drop = jump >= 2.2  # stronger jump = drop
                beats.append({
                    "time_seconds": round(time_sec, 3),
                    "frame_index": frame_idx,
                    "confidence": round(confidence, 3),
                    "is_drop": is_drop,
                    "drop_confidence": round(min(1.0, (jump - 1.0) / 2.0), 3) if is_drop else 0.0,
                    "is_downbeat": False,
                    "energy_jump": round(jump, 2),
                    "method": "rms",
                })

        # Merge near beats
        merged = []
        for b in sorted(beats, key=lambda x: x["time_seconds"]):
            if merged and (b["time_seconds"] - merged[-1]["time_seconds"]) < 0.5:
                if b["confidence"] > merged[-1]["confidence"]:
                    merged[-1] = b
            else:
                merged.append(b)

        limit = max(1, min(int(max_beats), 512))
        return (_json.dumps(merged[:limit]), len(merged[:limit]))

    def detect(self, audio=None, audio_path="", fps=30.0, video="", max_beats=64):
        fps = max(1.0, float(fps))
        max_beats = max(1, min(int(max_beats), 512))
        tmp_wav = None

        # Priority: AUDIO tensor > video > audio_path
        if audio is not None and isinstance(audio, dict) and "waveform" in audio:
            import torchaudio
            import tempfile
            waveform = audio["waveform"]
            sample_rate = audio["sample_rate"]
            # waveform shape: [channels, samples] or [1, channels, samples]
            if waveform.dim() == 3:
                waveform = waveform[0]
            if waveform.dim() == 2 and waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)  # mono mix
            tmp_wav = tempfile.mktemp(suffix=".wav", prefix="beatit_audio_")
            torchaudio.save(tmp_wav, waveform.cpu(), int(sample_rate))
            audio_path = tmp_wav

        # Video has priority: extract audio via ffmpeg
        if video and video.strip() and os.path.isfile(video.strip()):
            import tempfile
            tmp_wav = tempfile.mktemp(suffix=".wav", prefix="beatit_video_")
            try:
                subprocess.run([
                    "ffmpeg", "-hide_banner", "-loglevel", "error",
                    "-i", video.strip(),
                    "-vn", "-ac", "1", "-ar", "22050",
                    "-sample_fmt", "s16", "-y", tmp_wav,
                ], check=True, timeout=300, capture_output=True)
                audio_path = tmp_wav
            except subprocess.CalledProcessError as e:
                err = e.stderr.decode(errors="replace") if e.stderr else str(e)
                try:
                    os.unlink(tmp_wav)
                except Exception:
                    pass
                tmp_wav = None
                if "output file does not contain any stream" in err.lower():
                    print("[BeatIt] Video has no audio stream; returning an empty beat analysis.")
                    return ("[]", 0)
                raise RuntimeError(f"Failed to extract audio from video: {err[:500]}")

        try:
            if not audio_path or not os.path.isfile(audio_path):
                raise ValueError(f"Audio file not found: {audio_path or '(none)'}")

            # Try beat_this first
            try:
                beats = self._detect_with_beatthis(audio_path, fps)
                if beats:
                    return (_json.dumps(beats[:max_beats]), len(beats[:max_beats]))
            except Exception as exc:
                print(
                    "[BeatIt] beat_this failed; falling back to RMS: "
                    f"{type(exc).__name__}: {exc}"
                )

            # Fallback to RMS
            return self._detect_with_rms(audio_path, fps, max_beats=max_beats)
        finally:
            if tmp_wav:
                try:
                    os.unlink(tmp_wav)
                except Exception:
                    pass


def _make_blank_image(h=64, w=64):
    """Return a 1x1 black IMAGE tensor as placeholder."""
    return torch.zeros(1, h, w, 3)


class DuoSelectorNode:
    """Receives frame HTTP paths or IMAGE tensors, works in DUO with a Judge node.

    The Judge can send extra_penalty_json mapping frame URL → penalty value.
    These are ADDED on top of the standard Johnson history penalty — images
    that were recently selected get automatic decay, and the Judge can push
    additional penalty on specific rejected frames. Total penalty = Johnson
    history decay + extra_penalty from Judge.

    When beats_used (from FrameSequenceGenerator) is connected, frames are
    grouped per drop window and selection happens within each window. Without
    beats_used, all frames are treated as one flat pool (backward compat).

    Two modes:
      IMAGE mode: connect images (IMAGE tensor) — works with frame indices.
                  Outputs a contact_sheet of selected frames.
      URL mode:   connect http_paths (STRING) — works with HTTP URLs.
                  Backward compatible.

    Inputs:
      - http_paths: newline-separated frame URLs (optional if images connected)
      - images: IMAGE batch (optional — uses indices instead of URLs)
      - base_url: base URL for building HTTP paths from image indices
      - penalty: float 0-1 shift from Judge
      - extra_penalty_json: JSON dict {frame_url: penalty_value} from Judge
      - max_frames: int max candidates to forward (per window if beats_used)

    Outputs:
      - selected_paths: newline-separated HTTP URLs or frame indices
      - count: int
      - metadata: JSON with applied penalties + window grouping
      - system_prompt: passthrough from Judge
      - contact_sheet: IMAGE grid of selected frames (IMAGE mode only, else black)
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "penalty": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.1}),
                "max_frames": ("INT", {"default": 6, "min": 1, "max": 60, "tooltip": "Max frames to select (per window when beats_used is connected)"}),
            },
            "optional": {
                "http_paths": ("STRING", {"default": "", "multiline": True, "placeholder": "HTTP URLs from FrameSequenceGenerator"}),
                "images": ("IMAGE",),
                "base_url": ("STRING", {"default": "http://localhost:8188", "multiline": False, "placeholder": "Base URL for building HTTP paths from image indices"}),
                "endpoint": ("STRING", {"default": "", "multiline": False}),
                "extra_penalty_json": ("STRING", {"default": "{}", "multiline": True}),
                "beats_used": ("STRING", {"default": "", "multiline": True, "placeholder": "JSON from FrameSequenceGenerator beats_used output"}),
                "system_prompt": ("STRING", {"default": "", "multiline": True, "placeholder": "Extra instructions from Judge for the LLM Selector"}),
            },
        }

    RETURN_TYPES = ("STRING", "INT", "STRING", "STRING", "IMAGE")
    RETURN_NAMES = ("selected_paths", "count", "metadata", "system_prompt", "contact_sheet")
    FUNCTION = "select"
    CATEGORY = "Amin/Beatdrop"

    def select(self, penalty, max_frames, http_paths="", images=None, base_url="http://localhost:8188",
               endpoint="", extra_penalty_json="{}", beats_used="", system_prompt=""):
        # Determine mode: IMAGE mode if images tensor is provided
        image_mode = images is not None and isinstance(images, torch.Tensor)

        if image_mode:
            B = images.shape[0] if images is not None else 0
            paths = [str(i) for i in range(B)]  # use frame indices as identifiers
        else:
            paths = [p.strip() for p in (http_paths or "").split("\n") if p.strip()]

        if not paths:
            return ("", 0, "{}", system_prompt, _make_blank_image())

        # Parse extra penalties from Judge
        extra_penalties = {}
        try:
            extra_penalties = _json.loads(extra_penalty_json or "{}")
        except Exception:
            pass
        if not isinstance(extra_penalties, dict):
            extra_penalties = {}

        n = max(1, min(int(max_frames), len(paths)))
        penalty_shift = max(0.0, min(1.0, float(penalty)))

        def frame_score(path):
            score = 0.0
            for key, val in extra_penalties.items():
                if key in path:
                    score += float(val)
            return score

        # --- Parse beats_used for window grouping ---
        window_starts = []
        try:
            bu = _json.loads(beats_used or "[]")
            if isinstance(bu, list) and bu:
                for entry in bu:
                    window_starts.append((entry, 0))  # frame count computed below
        except Exception:
            bu = None

        if bu and window_starts:
            # WINDOW-AWARE mode: select per drop window
            all_selected = []
            window_results = []

            # Compute frame count per window proportional to duration
            total_duration = sum(
                max(0.001, float(e.get("range_end", 0)) - float(e.get("range_start", 0)))
                for e, _ in window_starts
            )
            for wi, (entry, _) in enumerate(window_starts):
                dur = max(0.001, float(entry.get("range_end", 0)) - float(entry.get("range_start", 0)))
                w_frames = max(1, int(round(len(paths) * dur / total_duration))) if total_duration > 0 else len(paths)
                window_starts[wi] = (entry, w_frames)

            # Distribute frames proportionally, ensuring total matches
            allocated = sum(wf for _, wf in window_starts)
            if allocated > len(paths):
                window_starts.sort(key=lambda x: -x[1])
                window_starts[0] = (window_starts[0][0], max(1, window_starts[0][1] - (allocated - len(paths))))
            elif allocated < len(paths):
                window_starts[-1] = (window_starts[-1][0], window_starts[-1][1] + (len(paths) - allocated))

            # Slice paths per window and select best frames within each
            cursor = 0
            for wi, (entry, w_frame_count) in enumerate(window_starts):
                w_end = min(len(paths), cursor + w_frame_count)
                w_paths = paths[cursor:w_end]
                cursor = w_end

                if not w_paths:
                    continue

                scored = [(p, frame_score(p)) for p in w_paths]
                scored.sort(key=lambda x: x[1])
                w_n = min(n, len(w_paths))
                w_selected = [p for p, _ in scored[:w_n]]

                all_selected.extend(w_selected)
                window_results.append({
                    "drop_index": wi,
                    "beat_time": entry.get("time_seconds"),
                    "frame_index": entry.get("frame_index"),
                    "is_drop": entry.get("is_drop", False),
                    "range_start": entry.get("range_start"),
                    "range_end": entry.get("range_end"),
                    "window_frames": len(w_paths),
                    "selected_count": len(w_selected),
                    "selected": w_selected,
                })

            selected = all_selected
            meta = _json.dumps({
                "mode": "window_aware",
                "total_frames": len(paths),
                "windows": len(window_results),
                "selected_count": len(selected),
                "penalty_shift": round(penalty_shift, 2),
                "extra_penalties_applied": len(extra_penalties),
                "window_results": window_results,
            })
        else:
            # FLAT mode: backward compat — select from all frames
            scored = [(p, frame_score(p)) for p in paths]
            scored.sort(key=lambda x: x[1])
            selected = [p for p, _ in scored[:n]]

            meta = _json.dumps({
                "mode": "flat",
                "total_frames": len(paths),
                "selected_count": len(selected),
                "penalty_shift": round(penalty_shift, 2),
                "selection_rule": "lowest_penalty_first",
                "extra_penalties_applied": len(extra_penalties),
                "scored_frames": [
                    {"path": p.split("/")[-1][:40], "penalty_score": round(s, 2)}
                    for p, s in scored[:n]
                ],
            })

        # Build contact sheet if in IMAGE mode
        contact_sheet = _make_blank_image()
        if image_mode and selected:
            try:
                sel_indices = [int(p) for p in selected]
                sel_frames = images[sel_indices]  # [K, H, W, C]
                # Arrange in a grid
                K = sel_frames.shape[0]
                cols = min(K, 4)
                rows = (K + cols - 1) // cols
                H, W = sel_frames.shape[1], sel_frames.shape[2]
                sheet = torch.zeros(rows * H, cols * W, 3)
                for ki in range(K):
                    r, c = ki // cols, ki % cols
                    sheet[r*H:(r+1)*H, c*W:(c+1)*W] = sel_frames[ki]
                contact_sheet = sheet.unsqueeze(0)  # [1, H*rows, W*cols, 3]
            except Exception:
                pass

        return ("\n".join(selected), len(selected), meta, system_prompt, contact_sheet)


class JudgeNode:
    """Judge outfit change quality — duo partner of DuoSelectorNode.

    Takes selected frame paths and decides whether the outfit change
    detection is correct. Returns a per-frame penalty_json that the
    DuoSelector can use as Johnson-style extra penalties on the next run.

    Outputs:
      - verdict_json: full judgment
      - penalty: float 0-1 shift
      - penalty_json: dict {frame_url: penalty_value} for DuoSelector
      - should_restart: bool
      - verdict_text: short human summary
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "selected_paths": ("STRING", {"default": "", "multiline": True}),
                "prompt": ("STRING", {"default": "Check if the outfit clearly changes between frames. Return JSON with approved(bool), score(0-100), reason(string).", "multiline": True}),
                "endpoint": ("STRING", {"default": "http://localhost:8080/v1/chat/completions", "multiline": False}),
                "model": ("STRING", {"default": "local-model", "multiline": False}),
                "api_token": ("STRING", {"default": "", "multiline": False}),
                "should_restart": ("BOOLEAN", {"default": False}),
                "rejected_frames": ("STRING", {"default": "", "multiline": True, "placeholder": "newline-separated frame URLs that were rejected"}),
                "base_penalty": ("FLOAT", {"default": 10.0, "min": 0.0, "max": 50.0, "step": 0.5, "tooltip": "Johnson penalty value per rejected frame"}),
            },
            "optional": {
                "reference_image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("STRING", "FLOAT", "STRING", "BOOLEAN", "STRING", "STRING")
    RETURN_NAMES = ("verdict_json", "penalty", "penalty_json", "should_restart", "verdict_text", "system_prompt")
    FUNCTION = "judge"
    CATEGORY = "Amin/Beatdrop"

    def judge(self, selected_paths, prompt, endpoint, model, api_token, should_restart, rejected_frames="", base_penalty=10.0, reference_image=None):
        paths = [p.strip() for p in (selected_paths or "").split("\n") if p.strip()]
        if not paths:
            return (
                _json.dumps({"approved": False, "score": 0, "reason": "no frames provided"}),
                1.0,
                "{}",
                True,
                "Keine Frames zum Prüfen.",
                "",
            )

        # Build per-frame penalty map from rejected frames
        rejected = [r.strip() for r in (rejected_frames or "").split("\n") if r.strip()]
        penalty_map = {}
        base = max(0.0, float(base_penalty))
        for i, url in enumerate(rejected):
            penalty_map[url] = round(base + i * 2.0, 1)

        verdict = {
            "approved": not should_restart,
            "score": 30 if should_restart else 80,
            "reason": f"judged {len(paths)} frames, rejected={len(rejected)}, restart={should_restart}",
            "frame_count": len(paths),
            "rejected_count": len(rejected),
            "penalty_map": {p.split("/")[-1][:30]: v for p, v in penalty_map.items()},
        }
        shift = 0.9 if should_restart else 0.0

        # System prompt for the LLM Selector — passed through from prompt input
        system_prompt = str(prompt or "").strip()

        return (
            _json.dumps(verdict, indent=2),
            shift,
            _json.dumps(penalty_map),
            should_restart,
            verdict["reason"],
            system_prompt,
        )


class AlphaRavisJudgeNode:
    """Thin same-thread AlphaRavis judge client.

    ComfyUI stays the renderer/analyzer. This node sends selected frame URLs,
    beat/drop metadata, and the current conversation/thread key to AlphaRavis.
    AlphaRavis can then use its own context and, if needed, delegate the visual
    judgment to a bounded subagent.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "selected_paths": ("STRING", {"default": "", "multiline": True}),
                "alpha_endpoint": ("STRING", {"default": "http://127.0.0.1:8123/v1/chat/completions", "multiline": False}),
                "model": ("STRING", {"default": "alpha_ravis", "multiline": False}),
                "api_token": ("STRING", {"default": "", "multiline": False}),
                "conversation_id": ("STRING", {"default": "", "multiline": False}),
                "instructions": ("STRING", {"default": "Judge these beatdrop/outfit-change candidate frames in the current AlphaRavis thread. Use a bounded visual subagent only if needed. Return strict JSON with approved, should_restart, rejected_frames, penalty_map, confidence, reason, selected_outfit_reference, last_old_outfit_frame, first_new_outfit_frame.", "multiline": True}),
                "job_policy": (["main_job_only", "every_job", "only_on_drop", "manual"], {"default": "main_job_only"}),
                "timeout": ("INT", {"default": 180, "min": 5, "max": 600, "step": 1}),
                "base_penalty": ("FLOAT", {"default": 10.0, "min": 0.0, "max": 100.0, "step": 0.5}),
            },
            "optional": {
                "thread_id": ("STRING", {"default": "", "multiline": False}),
                "run_id": ("STRING", {"default": "", "multiline": False}),
                "job_id": ("STRING", {"default": "", "multiline": False}),
                "drop_id": ("STRING", {"default": "", "multiline": False}),
                "beats_json": ("STRING", {"default": "", "multiline": True}),
                "local_paths": ("STRING", {"default": "", "multiline": True}),
                "context_json": ("STRING", {"default": "{}", "multiline": True}),
                "system_prompt": ("STRING", {"default": "You are AlphaRavis inside the user's existing thread. Preserve thread context, evaluate the supplied beatdrop/outfit-change evidence, and return strict JSON only.", "multiline": True}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "BOOLEAN", "STRING")
    RETURN_NAMES = ("verdict_json", "rejected_frames", "penalty_json", "should_restart", "raw_response")
    FUNCTION = "judge_with_alpharavis"
    CATEGORY = "Amin/Beatdrop"

    def judge_with_alpharavis(
        self,
        selected_paths,
        alpha_endpoint,
        model,
        api_token,
        conversation_id,
        instructions,
        job_policy,
        timeout,
        base_penalty,
        thread_id="",
        run_id="",
        job_id="",
        drop_id="",
        beats_json="",
        local_paths="",
        context_json="{}",
        system_prompt="",
    ):
        paths = [p.strip() for p in (selected_paths or "").split("\n") if p.strip()]
        locals_ = [p.strip() for p in (local_paths or "").split("\n") if p.strip()]
        thread_key = str(thread_id or conversation_id or "").strip()
        conv_key = str(conversation_id or thread_id or "").strip()

        if not paths:
            verdict = {"approved": False, "should_restart": True, "reason": "no selected_paths provided"}
            return (_json.dumps(verdict, ensure_ascii=False), "", "{}", True, "")
        if not alpha_endpoint or not str(alpha_endpoint).strip():
            raise ValueError("alpha_endpoint is required")
        if not thread_key and not conv_key:
            raise ValueError("conversation_id or thread_id is required for same-thread AlphaRavis judging")

        context = {}
        try:
            parsed_context = _json.loads(context_json or "{}")
            if isinstance(parsed_context, dict):
                context = parsed_context
        except Exception:
            context = {"raw_context_json": str(context_json or "")}

        beats = []
        try:
            parsed_beats = _json.loads(beats_json or "[]")
            if isinstance(parsed_beats, list):
                beats = parsed_beats
        except Exception:
            beats = []

        metadata = {
            "source": "comfyui_beatdrop",
            "node": "AlphaRavisJudgeNode",
            "conversation_id": conv_key,
            "thread_id": thread_key,
            "run_id": str(run_id or ""),
            "job_id": str(job_id or ""),
            "drop_id": str(drop_id or ""),
            "job_policy": str(job_policy or "manual"),
        }
        evidence = {
            "instructions": str(instructions or ""),
            "selected_paths": paths,
            "local_paths": locals_,
            "beats": beats,
            "context": context,
            "metadata": metadata,
            "expected_response_schema": {
                "approved": "boolean",
                "should_restart": "boolean",
                "rejected_frames": ["frame URL or filename substring"],
                "penalty_map": {"frame URL or filename substring": "number"},
                "confidence": "0..1 number",
                "reason": "string",
                "selected_outfit_reference": "string|null",
                "last_old_outfit_frame": "number|null",
                "first_new_outfit_frame": "number|null",
            },
        }

        headers = {"Content-Type": "application/json"}
        if api_token:
            headers["Authorization"] = f"Bearer {api_token}"
        if conv_key:
            headers["x-conversation-id"] = conv_key
        if thread_key:
            headers["x-thread-id"] = thread_key

        payload = {
            "model": model,
            "conversation_id": conv_key,
            "metadata": metadata,
            "messages": [
                {"role": "system", "content": str(system_prompt or "")},
                {"role": "user", "content": "AlphaRavis beatdrop judge request. Return strict JSON only:\n" + _json.dumps(evidence, ensure_ascii=False, indent=2)},
            ],
            "temperature": 0,
            "max_tokens": 1600,
            "stream": False,
        }

        raw, response_json = _post_json(str(alpha_endpoint).strip(), headers, payload, timeout)
        content = _extract_chat_content(response_json)
        verdict_obj = _extract_first_json_object(content)
        if not isinstance(verdict_obj, dict):
            verdict_obj = {"approved": False, "should_restart": True, "reason": "AlphaRavis response did not contain a JSON object", "raw_response": content}

        rejected = verdict_obj.get("rejected_frames", [])
        if isinstance(rejected, str):
            rejected_list = [r.strip() for r in rejected.replace(",", "\n").split("\n") if r.strip()]
        elif isinstance(rejected, list):
            rejected_list = [str(r).strip() for r in rejected if str(r).strip()]
        else:
            rejected_list = []

        penalty_map = verdict_obj.get("penalty_map") or verdict_obj.get("penalty_json") or {}
        if not isinstance(penalty_map, dict):
            penalty_map = {}
        base = max(0.0, float(base_penalty))
        for idx, frame in enumerate(rejected_list):
            penalty_map.setdefault(frame, round(base + idx * 2.0, 1))

        approved = bool(verdict_obj.get("approved", False))
        should_restart = bool(verdict_obj.get("should_restart", not approved))
        verdict_obj["approved"] = approved
        verdict_obj["should_restart"] = should_restart
        verdict_obj["rejected_frames"] = rejected_list
        verdict_obj["penalty_map"] = penalty_map
        verdict_obj.setdefault("metadata", metadata)

        return (
            _json.dumps(verdict_obj, ensure_ascii=False, indent=2),
            "\n".join(rejected_list),
            _json.dumps(penalty_map, ensure_ascii=False),
            should_restart,
            content or raw,
        )


# ComfyUI registration — at bottom after all class definitions
NODE_CLASS_MAPPINGS = {
    "FrameSequenceGenerator": FrameSequenceGenerator,
    "BeatItNode": BeatItNode,
    "DuoSelectorNode": DuoSelectorNode,
    "JudgeNode": JudgeNode,
    "AlphaRavisJudgeNode": AlphaRavisJudgeNode,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "FrameSequenceGenerator": "🎬 Frame Sequence Generator",
    "BeatItNode": "🔊 BeatIt (BeatThis + RMS Fallback)",
    "DuoSelectorNode": "👥 Duo Selector (API Style)",
    "JudgeNode": "⚖️ Judge (Outfit Check + Penalty)",
    "AlphaRavisJudgeNode": "🧠 AlphaRavis Judge (Same Thread)",
}