"""
DINOv2FrameChangeDetector — embedding-based outfit change detection.

Detects visual outfit changes in stabilized crop sequences using DINOv2 embeddings.
Computes cosine distance between consecutive frames, finds change peaks within a
search window around a beat frame, and supports three modes:

  1. detect_existing_change — finds strongest visual jump (for videos with changes)
  2. beat_only_generate_outfit_drop — plans a synthetic drop at beat frame (no video change)
  3. auto — detects existing change if found, otherwise plans synthetic drop

Supports alignment offsets to snap detected changes to beat timing.

Place in: comfyui-ex-rvc/frame_embedding_change_detector.py
"""

import json
import torch
import torch.nn.functional as F


# ── Model cache ───────────────────────────────────────────────────────

_DINOV2_CACHE = {}

# HuggingFace model IDs (primary download source)
DINOV2_MODELS = {
    "dinov2_vits14": {"dim": 384, "hf_id": "facebook/dinov2-small"},
    "dinov2_vitb14": {"dim": 768, "hf_id": "facebook/dinov2-base"},
    "dinov2_vitl14": {"dim": 1024, "hf_id": "facebook/dinov2-large"},
}


def _get_dinov2_model(model_name, device):
    """Lazy-load and cache DINOv2 model. Downloads automatically from HuggingFace Hub.

    Primary: transformers.AutoModel (HuggingFace Hub — auto-download, reliable)
    Fallback: torch.hub (GitHub CDN — may need auth)
    """
    key = (model_name, device)
    if key in _DINOV2_CACHE:
        return _DINOV2_CACHE[key]

    if model_name not in DINOV2_MODELS:
        raise ValueError(
            f"Unknown model '{model_name}'. Available: {list(DINOV2_MODELS.keys())}"
        )

    info = DINOV2_MODELS[model_name]
    hf_id = info["hf_id"]
    print(f"[DINOv2FrameChangeDetector] Loading model: {model_name}")
    print(f"[DINOv2FrameChangeDetector] HuggingFace ID: {hf_id}")
    print(f"[DINOv2FrameChangeDetector] Device: {device}")

    model = None
    errors = []

    # ── Primary: transformers (HuggingFace Hub auto-download) ──
    try:
        from transformers import AutoModel
        import os as _os

        cache_dir = _os.path.expanduser("~/.cache/huggingface/hub")
        print(f"[DINOv2FrameChangeDetector] Downloading from HuggingFace Hub...")
        print(f"[DINOv2FrameChangeDetector] Cache: {cache_dir}")

        model = AutoModel.from_pretrained(
            hf_id,
            trust_remote_code=False,
            local_files_only=False,  # Allow download
        )
        model.to(device)
        model.eval()
        _DINOV2_CACHE[key] = (model, info["dim"])
        print(f"[DINOv2FrameChangeDetector] Model loaded successfully via HuggingFace (dim={info['dim']})")
        return _DINOV2_CACHE[key]

    except Exception as e:
        errors.append(f"HuggingFace: {e}")
        print(f"[DINOv2FrameChangeDetector] HuggingFace load failed: {e}")

    # ── Fallback: torch.hub ──
    try:
        import torch
        print(f"[DINOv2FrameChangeDetector] Falling back to torch.hub...")
        model = torch.hub.load("facebookresearch/dinov2", model_name,
                               trust_repo=True, force_reload=False)
        model.to(device)
        model.eval()
        _DINOV2_CACHE[key] = (model, info["dim"])
        print(f"[DINOv2FrameChangeDetector] Model loaded successfully via torch.hub (dim={info['dim']})")
        return _DINOV2_CACHE[key]

    except Exception as e:
        errors.append(f"torch.hub: {e}")

    # ── Both failed ──
    raise RuntimeError(
        f"Failed to load DINOv2 model '{model_name}'.\n"
        f"Primary (HuggingFace): {errors[0]}\n"
        f"Fallback (torch.hub): {errors[1] if len(errors) > 1 else 'N/A'}\n\n"
        f"Troubleshooting:\n"
        f"  1. Check internet: curl -I https://huggingface.co\n"
        f"  2. Pre-download manually:\n"
        f"     python -c \"from transformers import AutoModel; AutoModel.from_pretrained('{hf_id}')\"\n"
        f"  3. Local cache path: ~/.cache/huggingface/hub/\n"
        f"  4. For offline use, download the model to a local folder and set HF_HOME.\n"
    )


# ── DINOv2FrameChangeDetector ─────────────────────────────────────────

class DINOv2FrameChangeDetector:
    """
    Detects outfit changes in a crop sequence using DINOv2 embeddings.

    Input: stabilized crops from MaskCropStabilizer
    Output: change JSON with best change frame, scores, alignment info,
            and mode-dependent fields (planned drop, existing change flag).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "crops": ("IMAGE", {"tooltip": "Stabilized crop sequence from MaskCropStabilizer"}),
                "mode": (["detect_existing_change", "beat_only_generate_outfit_drop", "auto"], {
                    "default": "auto",
                    "tooltip": "detect_existing_change: find real visual change. beat_only: plan synthetic drop. auto: either.",
                }),
                "model_name": (list(DINOV2_MODELS.keys()), {
                    "default": "dinov2_vitb14",
                    "tooltip": "DINOv2 variant. vitb14 = best quality/speed balance.",
                }),
                "device": (["auto", "cuda", "cpu"], {
                    "default": "auto",
                }),
                "change_threshold": ("FLOAT", {
                    "default": 0.25, "min": 0.05, "max": 1.0, "step": 0.01,
                    "tooltip": "Cosine distance threshold. Values above = possible outfit change.",
                }),
            },
            "optional": {
                "fps": ("FLOAT", {
                    "default": 30.0, "min": 1.0, "max": 120.0, "step": 1.0,
                }),
                "start_frame": ("INT", {
                    "default": 0, "min": 0, "max": 1000000, "step": 1,
                    "tooltip": "First frame number in the crop sequence",
                }),
                "beat_frame": ("INT", {
                    "default": 0, "min": 0, "max": 1000000, "step": 1,
                    "tooltip": "Beat frame from FrameSequenceGenerator/BeatItNode",
                }),
                "beat_time_sec": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 3600.0, "step": 0.1,
                }),
                "search_before_frames": ("INT", {
                    "default": 6, "min": 0, "max": 300, "step": 1,
                    "tooltip": "Frames BEFORE beat_frame to search for change",
                }),
                "search_after_frames": ("INT", {
                    "default": 12, "min": 0, "max": 300, "step": 1,
                    "tooltip": "Frames AFTER beat_frame to search for change",
                }),
                "top_k": ("INT", {
                    "default": 5, "min": 1, "max": 50, "step": 1,
                    "tooltip": "How many top change candidates to report",
                }),
                "alignment_offset_frames": ("INT", {
                    "default": 0, "min": -60, "max": 60, "step": 1,
                    "tooltip": "Systematic offset: e.g., +2 if visual change is consistently 2 frames after beat",
                }),
                "manual_offset_frames": ("INT", {
                    "default": 0, "min": -300, "max": 300, "step": 1,
                    "tooltip": "Manual per-video offset override",
                }),
            },
        }

    RETURN_TYPES = (
        "STRING",   # change_json
        "INT",      # best_change_frame
        "INT",      # last_old_outfit_frame
        "INT",      # first_new_outfit_frame
        "FLOAT",    # confidence
        "BOOLEAN",  # has_existing_visual_change
        "BOOLEAN",  # needs_generated_outfit_drop
        "STRING",   # report
    )
    RETURN_NAMES = (
        "change_json",
        "best_change_frame",
        "last_old_outfit_frame",
        "first_new_outfit_frame",
        "confidence",
        "has_existing_visual_change",
        "needs_generated_outfit_drop",
        "report",
    )
    FUNCTION = "detect"
    CATEGORY = "Amin/Researcher"

    # ── Embedding computation ─────────────────────────────────────────

    def _compute_embeddings(self, crops_tensor, model_name, device):
        """Compute DINOv2 embeddings for each crop. Returns (B, dim) tensor."""
        model, dim = _get_dinov2_model(model_name, device)
        B = crops_tensor.shape[0]

        imgs = crops_tensor.permute(0, 3, 1, 2)  # BHWC → BCHW
        imgs = F.interpolate(imgs, size=(224, 224), mode="bilinear", align_corners=False)

        embeddings = []
        batch_size = 16
        for i in range(0, B, batch_size):
            batch = imgs[i : i + batch_size].to(device)
            with torch.no_grad():
                out = model(batch)
            # Extract embeddings — supports both transformers (BaseModelOutputWithPooling)
            # and raw torch.hub models (returns tensor directly)
            if hasattr(out, "pooler_output"):
                emb = out.pooler_output
            elif hasattr(out, "last_hidden_state"):
                # CLS token from last hidden state (position 0)
                emb = out.last_hidden_state[:, 0, :]
            elif isinstance(out, torch.Tensor):
                emb = out
            else:
                raise TypeError(f"Unexpected model output type: {type(out)}. Expected tensor or BaseModelOutputWithPooling.")
            embeddings.append(emb.cpu())
            del batch

        return torch.cat(embeddings, dim=0)

    # ── Main detect ───────────────────────────────────────────────────

    def detect(
        self,
        crops,
        mode,
        model_name,
        device,
        change_threshold,
        fps=30.0,
        start_frame=0,
        beat_frame=0,
        beat_time_sec=0.0,
        search_before_frames=6,
        search_after_frames=12,
        top_k=5,
        alignment_offset_frames=0,
        manual_offset_frames=0,
    ):
        # ── Validate ──
        if crops is None or not isinstance(crops, torch.Tensor):
            return self._empty_result("No crops provided")

        B = crops.shape[0]
        if B < 2:
            return self._empty_result(f"Need at least 2 crops, got {B}")

        # Resolve device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        start_frame = int(start_frame)
        beat_frame = int(beat_frame)
        search_before = int(search_before_frames)
        search_after = int(search_after_frames)

        # ── Compute embeddings and pairwise distances ──
        embeddings = self._compute_embeddings(crops, model_name, device)
        embeddings = F.normalize(embeddings, p=2, dim=1)

        # Cosine distance: 1 - dot(A, B)
        similarities = (embeddings[:-1] * embeddings[1:]).sum(dim=1)
        distances = 1.0 - similarities  # (B-1,)
        dist_np = distances.numpy()

        # Map to absolute frame numbers
        # distance[i] = difference between frame (start_frame + i) and (start_frame + i + 1)
        frame_pairs = [
            {
                "from_frame": int(start_frame + i),
                "to_frame": int(start_frame + i + 1),
                "cosine_distance": round(float(dist_np[i]), 6),
            }
            for i in range(len(dist_np))
        ]

        # ── Define search window ──
        sw_start = max(0, beat_frame - search_before)
        sw_end = min(B + start_frame, beat_frame + search_after)

        # Map to internal indices
        win_pairs = [
            (i, fp)
            for i, fp in enumerate(frame_pairs)
            if sw_start <= fp["from_frame"] < sw_end
        ]

        # ── Find top changes within window ──
        scored = sorted(
            [(i, fp["cosine_distance"]) for i, fp in win_pairs],
            key=lambda x: x[1],
            reverse=True,
        )
        top_changes = []
        for idx, score in scored[: int(top_k)]:
            fp = frame_pairs[idx]
            top_changes.append({
                "from_frame": fp["from_frame"],
                "to_frame": fp["to_frame"],
                "score": round(score, 6),
                "above_threshold": score >= change_threshold,
            })

        has_strong_change = scored and scored[0][1] >= change_threshold

        # ── Mode-specific logic ──
        if mode == "beat_only_generate_outfit_drop":
            # ALWAYS plan synthetic drop, regardless of what we find
            has_existing = False
            needs_generated = True
        elif mode == "detect_existing_change":
            # ONLY detect existing, never generate
            has_existing = has_strong_change
            needs_generated = False
        else:  # "auto"
            has_existing = has_strong_change
            needs_generated = not has_strong_change

        # ── Determine change frame ──
        if has_existing and top_changes:
            best = top_changes[0]
            detected_change_frame = best["to_frame"]
            best_score = best["score"]
            confidence = min(1.0, max(0.0, (best_score - change_threshold * 0.5) / max(1.0 - change_threshold * 0.5, 0.01)))
        else:
            top_changes = top_changes[:1] if top_changes else []
            best_score = top_changes[0]["score"] if top_changes else 0.0
            detected_change_frame = None
            confidence = 0.0  # No detected visual change; DINO is not a drop authority.

        # ── Alignment ──
        requested_total_offset = int(alignment_offset_frames) + int(manual_offset_frames)
        applied_offset = requested_total_offset if has_existing else 0

        if has_existing and detected_change_frame is not None:
            aligned_drop_frame = detected_change_frame + applied_offset
            last_old = detected_change_frame
            first_new = detected_change_frame
        elif needs_generated:
            # No visual change: the audio beat remains authoritative and visual
            # alignment offsets must not move it.
            aligned_drop_frame = beat_frame
            last_old = beat_frame - 1
            first_new = beat_frame
            detected_change_frame = None
        else:
            # No change and no generation (detect_existing_change mode, no strong change)
            aligned_drop_frame = beat_frame
            last_old = beat_frame
            first_new = beat_frame
            detected_change_frame = None

        # ── Build output JSON ──
        result = {
            "schema_version": 1,
            "node": "DINOv2FrameChangeDetector",
            "model": model_name,
            "device": device,
            "mode": mode,
            "beat_frame": beat_frame,
            "beat_time_sec": round(float(beat_time_sec), 2),
            "search_window": {
                "start_frame": sw_start,
                "end_frame": sw_end,
            },
            "best_change": {
                "from_frame": detected_change_frame - 1 if detected_change_frame else None,
                "to_frame": detected_change_frame,
                "score": round(best_score, 6),
            } if detected_change_frame else None,
            "top_changes": top_changes,
            "alignment": {
                "alignment_offset_frames": int(alignment_offset_frames),
                "manual_offset_frames": int(manual_offset_frames),
                "requested_total_offset": requested_total_offset,
                "total_offset": applied_offset,
                "final_drop_frame": aligned_drop_frame,
            },
            "has_existing_visual_change": has_existing,
            "needs_generated_outfit_drop": needs_generated,
            "dino_used_for_drop_decision": bool(has_existing),
            "drop_decision_source": (
                "dinov2_visual_change" if has_existing else "audio_beat"
            ),
            "last_old_outfit_frame": last_old,
            "first_new_outfit_frame": first_new,
            "confidence": round(confidence, 4),
        }

        if needs_generated and not has_existing:
            result["reason"] = (
                "No visual outfit change above threshold; DINOv2 is ignored for "
                f"drop timing and the audio beat at frame {beat_frame} remains authoritative."
            )
            result["dino_ignored_reason"] = "no_existing_visual_outfit_change"

        # Report string
        report_lines = [
            f"[DINOv2FrameChangeDetector] Mode: {mode}",
            f"  Model: {model_name} | Device: {device}",
            f"  Beat frame: {beat_frame} | Search: [{sw_start}, {sw_end}]",
            f"  Change threshold: {change_threshold}",
        ]
        if has_existing:
            report_lines.append(
                f"  [FOUND] Visual change at frame {detected_change_frame} "
                f"(score={best_score:.4f})"
            )
        else:
            report_lines.append(f"  [NONE] No visual change above threshold (best={best_score:.4f})")
        if needs_generated:
            report_lines.append(
                f"  [AUDIO] DINO ignored for timing; audio beat frame {aligned_drop_frame} is authoritative"
            )
        report_lines.append(
            f"  last_old={last_old} first_new={first_new} "
            f"has_existing={has_existing} needs_gen={needs_generated} conf={confidence:.3f}"
        )
        report_str = "\n".join(report_lines)

        return (
            json.dumps(result, indent=2),
            int(aligned_drop_frame),
            int(last_old),
            int(first_new),
            float(confidence),
            bool(has_existing),
            bool(needs_generated),
            report_str,
        )

    def _empty_result(self, reason):
        """Return a valid empty result when no crops available."""
        empty_json = json.dumps({
            "schema_version": 1,
            "node": "DINOv2FrameChangeDetector",
            "error": reason,
            "has_existing_visual_change": False,
            "needs_generated_outfit_drop": False,
        }, indent=2)
        return (empty_json, -1, -1, -1, 0.0, False, False, f"[DINOv2FrameChangeDetector] ERROR: {reason}")


# ── Node registration ─────────────────────────────────────────────────

NODE_CLASS_MAPPINGS = {
    "DINOv2FrameChangeDetector": DINOv2FrameChangeDetector,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DINOv2FrameChangeDetector": "DINOv2 Frame Change Detector (Outfit)",
}
