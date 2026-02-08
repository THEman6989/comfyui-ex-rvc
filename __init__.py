import os
import subprocess
import torch
import torchaudio
import tempfile
import shutil
import folder_paths

class RVC_Terminal_Node:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        base_path = folder_paths.base_path

        # --- QUELLE: ComfyUI Ordner ---
        # Pfade GROSS geschrieben wie besprochen
        rvc_models_dir = os.path.join(base_path, "models", "RVC")
        rvc_index_dir = os.path.join(base_path, "models", "RVC", ".index")

        os.makedirs(rvc_models_dir, exist_ok=True)
        os.makedirs(rvc_index_dir, exist_ok=True)

        # Modelle scannen
        if os.path.exists(rvc_models_dir):
            model_files = [f for f in os.listdir(rvc_models_dir) if f.endswith(".pth")]
        else:
            model_files = []

        if not model_files:
            model_files = ["Keine Modelle in models/RVC gefunden!"]

        # Indices scannen
        if os.path.exists(rvc_index_dir):
            index_files = [f for f in os.listdir(rvc_index_dir) if f.endswith(".index")]
        else:
            index_files = []

        index_files = ["None"] + index_files

        return {
            "required": {
                "audio": ("AUDIO",),
                "model_name": (sorted(model_files), ),
                "index_path": (sorted(index_files), ),

                "f0up_key": ("INT", {"default": 0, "min": -24, "max": 24, "step": 1, "display": "number"}),
                "f0method": (["rmvpe", "pm", "harvest"], {"default": "rmvpe"}),
                "index_rate": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.01}),
                "filter_radius": ("INT", {"default": 3, "min": 0, "max": 7}),
                "rms_mix_rate": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01}),
                "protect": ("FLOAT", {"default": 0.33, "min": 0.0, "max": 0.5, "step": 0.01}),
                # Dein Basis-Pfad
                "rvc_base_path": ("STRING", {"default": "/home/amin/experi/Retrieval-based-Voice-Conversion-WebUI/"}),
                "gpu_id": ("INT", {"default": 1, "min": 0, "max": 8}),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "process_rvc"
    CATEGORY = "Amin/RVC"

    def process_rvc(self, audio, model_name, f0up_key, f0method, index_path, index_rate, filter_radius, rms_mix_rate, protect, rvc_base_path, gpu_id):

        if model_name == "Keine Modelle in models/RVC gefunden!":
             raise FileNotFoundError("Bitte .pth Datei in ComfyUI/models/RVC/ legen.")

        base_path = folder_paths.base_path

        # --- 1. MODELL HANDLER ---
        # Quelle (Comfy)
        source_model_path = os.path.join(base_path, "models", "RVC", model_name)

        # Ziel (RVC Internal: assets/weights)
        rvc_weights_dir = os.path.join(rvc_base_path, "assets", "weights")
        # Fallback falls Ordnerstruktur anders ist
        if not os.path.exists(rvc_weights_dir):
             rvc_weights_dir = os.path.join(rvc_base_path, "weights")

        dest_model_path = os.path.join(rvc_weights_dir, model_name)

        # Symlink erstellen: Model
        if not os.path.exists(dest_model_path):
            print(f"Erstelle Symlink für Model: {dest_model_path} -> {source_model_path}")
            try:
                os.symlink(source_model_path, dest_model_path)
            except OSError as e:
                print(f"Symlink fehlgeschlagen, kopiere: {e}")
                shutil.copy2(source_model_path, dest_model_path)

        # --- 2. INDEX HANDLER ---
        final_index_arg = "" # Standard leer

        if index_path != "None" and index_path is not None:
            # Quelle (Comfy)
            source_index_path = os.path.join(base_path, "models", "RVC", ".index", index_path)

            # Ziel (RVC Internal: logs/)
            rvc_logs_dir = os.path.join(rvc_base_path, "logs")
            if not os.path.exists(rvc_logs_dir):
                os.makedirs(rvc_logs_dir, exist_ok=True)

            dest_index_path = os.path.join(rvc_logs_dir, index_path)

            # Symlink erstellen: Index
            if not os.path.exists(dest_index_path):
                print(f"Erstelle Symlink für Index: {dest_index_path} -> {source_index_path}")
                try:
                    os.symlink(source_index_path, dest_index_path)
                except OSError as e:
                    print(f"Symlink fehlgeschlagen, kopiere: {e}")
                    shutil.copy2(source_index_path, dest_index_path)

            # RVC Argument setzen (voller Pfad zur Datei im logs Ordner)
            final_index_arg = dest_index_path
        else:
            # Wenn kein Index gewählt, Rate auf 0
            index_rate = 0


        # --- 3. AUDIO VORBEREITUNG ---
        waveform = audio['waveform']
        sample_rate = audio['sample_rate']
        if waveform.dim() == 3:
            current_waveform = waveform[0]
        else:
            current_waveform = waveform

        temp_dir = tempfile.mkdtemp(prefix="comfy_rvc_")
        input_wav_path = os.path.join(temp_dir, "temp_input.wav")
        output_wav_path = os.path.join(temp_dir, "output_converted.wav")

        try:
            torchaudio.save(input_wav_path, current_waveform, sample_rate)

            venv_python = os.path.join(rvc_base_path, "venv/bin/python")
            script_path = os.path.join(rvc_base_path, "tools/infer_cli.py")

            if not os.path.exists(venv_python):
                 raise FileNotFoundError(f"Python Venv nicht gefunden: {venv_python}")

            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

            # Befehl bauen
            command = [
                venv_python, script_path,
                "--model_name", model_name,  # Nur Dateiname! RVC sucht automatisch in assets/weights
                "--input_path", input_wav_path,
                "--opt_path", output_wav_path,
                "--f0up_key", str(f0up_key),
                "--f0method", f0method,
                "--index_rate", str(index_rate),
                "--device", "cuda:0",
                "--filter_radius", str(filter_radius),
                "--resample_sr", "0",
                "--rms_mix_rate", str(rms_mix_rate),
                "--protect", str(protect)
            ]

            # Index hinzufügen, falls vorhanden
            if final_index_arg:
                command.extend(["--index_path", final_index_arg])

            print(f"RVC Executing: {' '.join(command)}")

            result = subprocess.run(
                command,
                cwd=rvc_base_path,
                capture_output=True,
                text=True,
                env=env
            )

            if result.returncode != 0:
                print("RVC Error Output:", result.stderr)
                print("RVC Std Output:", result.stdout)
                raise RuntimeError("RVC Process failed. Siehe Konsole für Details.")

            if os.path.exists(output_wav_path):
                out_waveform, out_sample_rate = torchaudio.load(output_wav_path)
                out_waveform = out_waveform.unsqueeze(0)
                result_audio = {"waveform": out_waveform, "sample_rate": out_sample_rate}
            else:
                raise FileNotFoundError("RVC Output-Datei wurde nicht erstellt.")

        finally:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)

        return (result_audio,)


import torch
import numpy as np
import os
from PIL import Image, ImageOps
import folder_paths

# --- HILFSFUNKTIONEN ---

def tensor2pil(image):
    # Konvertiert Tensor Batch zu PIL Liste
    batch_count = image.size(0) if len(image.shape) > 3 else 1
    if batch_count > 1:
        out = []
        for i in range(batch_count):
            out.append(Image.fromarray(np.clip(255. * image[i].cpu().numpy(), 0, 255).astype(np.uint8)))
        return out
    return [Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))]

def pil2tensor(image):
    # Konvertiert PIL Liste zurück zu Tensor Batch
    if isinstance(image, list):
        out = []
        for img in image:
            out.append(torch.from_numpy(np.array(img).astype(np.float32) / 255.0).unsqueeze(0))
        return torch.cat(out, dim=0)
    
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

# --- NODES ---

class Standalone_OverlayTransparentImage:
    
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                "back_image": ("IMAGE",),
                "overlay_image": ("IMAGE",),
                "transparency": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "offset_x": ("INT", {"default": 0, "min": -4096, "max": 4096}),
                "offset_y": ("INT", {"default": 0, "min": -4096, "max": 4096}),
                "rotation_angle": ("FLOAT", {"default": 0.0, "min": -360.0, "max": 360.0, "step": 0.1}),
                "overlay_scale_factor": ("FLOAT", {"default": 1.000, "min": 0.000, "max": 100.000, "step": 0.001}),
                }        
        }

    RETURN_TYPES = ("IMAGE", )
    FUNCTION = "overlay_image"
    CATEGORY = "Standalone/Graphics"

    def overlay_image(self, back_image, overlay_image, 
                      transparency, offset_x, offset_y, rotation_angle, overlay_scale_factor=1.0):

        # Konvertiere Input Tensoren zu PIL Listen
        back_images_pil = tensor2pil(back_image)
        overlay_images_pil = tensor2pil(overlay_image)
        
        results = []

        # Iteriere über die Hintergrund-Bilder (Batch/Video Support)
        for i, bg_img in enumerate(back_images_pil):
            
            # Wähle das passende Overlay Bild (loopen, falls weniger Overlays als Hintergründe)
            if i < len(overlay_images_pil):
                ov_img = overlay_images_pil[i]
            else:
                ov_img = overlay_images_pil[-1] if len(overlay_images_pil) > 0 else overlay_images_pil[0]

            # Arbeitskopien erstellen
            current_overlay = ov_img.copy()
            current_bg = bg_img.copy()

            # 1. Overlay immer in RGBA wandeln für korrekte Alpha-Verarbeitung
            if current_overlay.mode != 'RGBA':
                current_overlay = current_overlay.convert('RGBA')

            # 2. Transparenz anwenden (FIX: Werte ändern, nicht Größe)
            if transparency > 0.0:
                # Hole den Alpha-Kanal
                alpha = current_overlay.split()[3]
                # Berechne Faktor (0.0 = transparent, 1.0 = deckend)
                factor = 1.0 - transparency
                # Multipliziere jeden Pixel im Alpha-Kanal mit dem Faktor
                alpha = alpha.point(lambda p: int(p * factor))
                # Setze den modifizierten Alpha-Kanal zurück ins Bild
                current_overlay.putalpha(alpha)

            # 3. Rotation
            if rotation_angle != 0:
                current_overlay = current_overlay.rotate(rotation_angle, expand=True)

            # 4. Skalierung
            if overlay_scale_factor != 1.0:
                overlay_width, overlay_height = current_overlay.size
                new_size = (int(overlay_width * overlay_scale_factor), int(overlay_height * overlay_scale_factor))
                if new_size[0] > 0 and new_size[1] > 0:
                    current_overlay = current_overlay.resize(new_size, Image.Resampling.LANCZOS)

            # 5. Positionierung (zentriert + offset)
            center_x = current_bg.width // 2
            center_y = current_bg.height // 2
            position_x = center_x - current_overlay.width // 2 + offset_x
            position_y = center_y - current_overlay.height // 2 + offset_y

            # 6. Einfügen (Pasting)
            # Hintergrund temporär zu RGBA, damit Alpha-Blending sauber funktioniert
            if current_bg.mode != 'RGBA':
                current_bg = current_bg.convert('RGBA')
                
            # Paste overlay: Nutzt den Alpha-Kanal des Overlays als Maske für weiches Blending
            current_bg.paste(current_overlay, (position_x, position_y), mask=current_overlay)
            
            # Ergebnis zurück zu RGB konvertieren
            results.append(current_bg.convert('RGB'))

        # Liste von PIL Bildern zurück zu Tensor Batch
        return (pil2tensor(results),)


class Standalone_SaveImageClean:
    """
    Speichert Bilder OHNE Metadaten (kein Workflow JSON, kein EXIF).
    """
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = ""

    @classmethod
    def INPUT_TYPES(s):
        return {"required": 
                    {"images": ("IMAGE", ),
                     "filename_prefix": ("STRING", {"default": "clean_image"})},
                }

    RETURN_TYPES = ()
    FUNCTION = "save_images_clean"
    OUTPUT_NODE = True
    CATEGORY = "Standalone/IO"

    def save_images_clean(self, images, filename_prefix="clean_image"):
        filename_prefix += self.prefix_append
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, self.output_dir, images[0].shape[1], images[0].shape[0])
        
        results = list()
        
        # Konvertieren zu PIL
        pil_images = tensor2pil(images)

        for image in pil_images:
            file = f"{filename}_{counter:05}_.png"
            
            # Ein neues Bild erstellen, um sicherzugehen, dass keine 'info' kopiert wird
            clean_image = Image.new(image.mode, image.size)
            clean_image.putdata(image.getdata())
            
            # Speichern ohne pnginfo Parameter -> Keine Metadaten
            clean_image.save(os.path.join(full_output_folder, file), format="PNG", compress_level=4)
            
            results.append({
                "filename": file,
                "subfolder": subfolder,
                "type": self.type
            })
            counter += 1

        return { "ui": { "images": results } }

# --- MAPPINGS ---
import torch

class VAEDtypeChecker:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae": ("VAE",),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("dtype_text", "device_text")
    FUNCTION = "check_vae"
    CATEGORY = "utils"

    def check_vae(self, vae):
        # Wir erstellen ein minimales Testbild (1x1 Pixel), um die VAE zu triggern
        test_image = torch.zeros((1, 64, 64, 3))
        
        try:
            # Wir kodieren das Bild, um zu sehen, welchen Datentyp die VAE ausgibt
            latent_test = vae.encode(test_image)
            
            # In ComfyUI geben VAEs oft direkt den Tensor oder ein Dict zurück
            if isinstance(latent_test, dict):
                dtype = latent_test["samples"].dtype
                device = latent_test["samples"].device
            else:
                dtype = latent_test.dtype
                device = latent_test.device
                
            dtype_str = str(dtype)
            device_str = str(device)
            
        except Exception as e:
            dtype_str = f"Fehler beim Auslesen: {str(e)}"
            device_str = "unbekannt"

        print(f"VAE Check - Dtype: {dtype_str}, Device: {device_str}")
        return (dtype_str, device_str)


class WanVideoSeamBlender:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "base_images": ("IMAGE",),       # Der erste Clip
                "overlap_images": ("IMAGE",),    # Der zweite Clip (mit dem Overlap am Anfang)
                "overlap_length": ("INT", {"default": 5, "min": 1, "max": 100, "step": 1}),
                "interpolation": (["linear", "sigmoid"], {"default": "linear"}), # Linear oder weicher Kurvenverlauf
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("joined_images",)
    FUNCTION = "blend_video_seams"
    CATEGORY = "Amin/Video"

    def blend_video_seams(self, base_images, overlap_images, overlap_length, interpolation):
        # Sicherheitscheck: Batch-Größe prüfen
        frames_base = base_images.shape[0]
        frames_overlap = overlap_images.shape[0]

        # Falls der Overlap größer als die Videos ist, korrigieren wir das
        actual_overlap = min(overlap_length, frames_base, frames_overlap)
        
        if actual_overlap == 0:
            # Einfach aneinanderhängen wenn kein Overlap
            return (torch.cat((base_images, overlap_images), dim=0),)

        # 1. Teile definieren, die NICHT angefasst werden
        # Alles vom ersten Video AUSSER den letzten 'overlap' Frames
        part_1_clean = base_images[:-actual_overlap]
        
        # Alles vom zweiten Video AUSSER den ersten 'overlap' Frames
        part_3_clean = overlap_images[actual_overlap:]

        # 2. Die Bereiche holen, die gemischt werden sollen
        # Letzte X Frames von Base
        blend_chunk_base = base_images[-actual_overlap:]
        # Erste X Frames von Overlap
        blend_chunk_new = overlap_images[:actual_overlap]

        # 3. Blending Berechnung
        blended_frames = []
        
        for i in range(actual_overlap):
            # Berechne den Fortschritt (Alpha) von 0.0 bis 1.0
            # Bei Linear und overlap 5: ca. 0.16, 0.33, 0.5, 0.66, 0.83
            progress = (i + 1) / (actual_overlap + 1)

            if interpolation == "sigmoid":
                # Weichere Kurve (S-Kurve), damit der Übergang nicht so hart startet/endet
                # Einfache Sigmoid-Annäherung: 3x^2 - 2x^3 (Smoothstep)
                alpha = progress * progress * (3 - 2 * progress)
            else:
                # Linear
                alpha = progress

            # Mischen: (Base * (1-alpha)) + (New * alpha)
            # Wir nutzen torch.lerp für effizientes Mischen auf der GPU/CPU
            frame_base = blend_chunk_base[i]
            frame_new = blend_chunk_new[i]
            
            # Formel: input + weight * (end - input) -> Base + alpha * (New - Base)
            blended_frame = torch.lerp(frame_base, frame_new, alpha)
            
            # Dimension (1, H, W, C) für das Zusammenfügen wiederherstellen
            blended_frames.append(blended_frame.unsqueeze(0))

        # Liste der geblendeten Frames in einen Tensor umwandeln
        part_2_blended = torch.cat(blended_frames, dim=0)

        # 4. Alles zusammenfügen
        # Wenn part_1_clean leer ist (z.B. wenn Video kürzer als Overlap war), beachten
        parts = []
        if part_1_clean.shape[0] > 0:
            parts.append(part_1_clean)
        
        parts.append(part_2_blended)
        
        if part_3_clean.shape[0] > 0:
            parts.append(part_3_clean)

        result = torch.cat(parts, dim=0)

        return (result,)


class WanVideoSeamCC:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "base_images": ("IMAGE",),       # Clip A (wird am Ende gekürzt)
                "overlap_images": ("IMAGE",),    # Clip B (wird am Anfang farbkorrigiert)
                "overlap_length": ("INT", {"default": 5, "min": 0, "max": 100, "step": 1}),
                "method": (["mkl", "hm", "reinhard", "mvgd"], {"default": "mkl"}),
                "match_strength_start": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "match_strength_end": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "interpolation": (["linear", "sigmoid"], {"default": "linear"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("joined_video",)
    FUNCTION = "join_with_cc"
    CATEGORY = "Amin/Video"

    def join_with_cc(self, base_images, overlap_images, overlap_length, method, match_strength_start, match_strength_end, interpolation):
        try:
            from color_matcher import ColorMatcher
        except ImportError:
            print("WanSeamCC: 'color_matcher' fehlt. Bitte installieren.")
            # Fallback: Einfach hart zusammenfügen
            return (torch.cat((base_images, overlap_images), dim=0),)

        # 1. Berechne den tatsächlichen Overlap
        frames_base = base_images.shape[0]
        frames_new = overlap_images.shape[0]
        actual_overlap = min(overlap_length, frames_base, frames_new)

        if actual_overlap == 0:
            # Kein Overlap -> Einfach aneinanderhängen
            return (torch.cat((base_images, overlap_images), dim=0),)

        # 2. Videos zerlegen
        # ALTES VIDEO: Wir behalten alles BIS auf die letzten 'overlap' Frames
        # Die letzten Frames dienen nur als Farbreferenz und werden dann "weggeworfen"
        part_1_clean = base_images[:-actual_overlap] 
        ref_chunk = base_images[-actual_overlap:]    # Die "weggeworfenen" Frames (Referenz)

        # NEUES VIDEO: Wir trennen den Overlap-Teil ab, um ihn zu bearbeiten
        target_chunk = overlap_images[:actual_overlap].clone() # Wird korrigiert
        part_2_clean = overlap_images[actual_overlap:]         # Der Rest bleibt original

        # 3. Color Matching Loop für den Overlap-Teil
        cm = ColorMatcher()
        print(f"WanSeamCC: Verbinde Videos mit {actual_overlap} Frames Overlap-Korrektur.")

        for i in range(actual_overlap):
            # Ramp Berechnung
            if actual_overlap > 1:
                raw_progress = i / (actual_overlap - 1)
            else:
                raw_progress = 0.0

            if interpolation == "sigmoid":
                progress = raw_progress * raw_progress * (3 - 2 * raw_progress)
            else:
                progress = raw_progress

            # Stärke interpolieren
            current_strength = (1.0 - progress) * match_strength_start + progress * match_strength_end

            if current_strength <= 0.001:
                continue

            # 1:1 Matching (Frame i von Neu auf Frame i von Alt-Referenz)
            src_img = target_chunk[i].cpu().numpy()
            ref_img = ref_chunk[i].cpu().numpy()

            try:
                matched_data = cm.transfer(src=src_img, ref=ref_img, method=method)
                matched_tensor = torch.from_numpy(matched_data).to(target_chunk.device)
                
                # Mischen
                target_chunk[i] = torch.lerp(target_chunk[i], matched_tensor, current_strength)
            
            except Exception as e:
                print(f"Fehler Frame {i}: {e}")

        # 4. Alles zusammenfügen (Concatenate)
        # [Altes Video Clean] + [Korrigierter Overlap] + [Neues Video Rest]
        
        parts = []
        if part_1_clean.shape[0] > 0:
            parts.append(part_1_clean)
        
        parts.append(target_chunk)
        
        if part_2_clean.shape[0] > 0:
            parts.append(part_2_clean)

        result = torch.cat(parts, dim=0)

        return (result,)


class WanVideoSeamCC_v2:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "base_images": ("IMAGE",),       # Clip A (wird am Ende korrigiert)
                "overlap_images": ("IMAGE",),    # Clip B (dient als Referenz, Anfang wird weggeschnitten)
                "overlap_length": ("INT", {"default": 5, "min": 0, "max": 100, "step": 1}),
                "method": (["mkl", "hm", "reinhard", "mvgd"], {"default": "mkl"}),
                # Standard hier umgekehrt: Von 0 (Original A) zu 1 (Match B)
                "match_strength_start": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "match_strength_end": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "interpolation": (["linear", "sigmoid"], {"default": "linear"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("joined_video",)
    FUNCTION = "join_with_cc_v2"
    CATEGORY = "Amin/Video"

    def join_with_cc_v2(self, base_images, overlap_images, overlap_length, method, match_strength_start, match_strength_end, interpolation):
        try:
            from color_matcher import ColorMatcher
        except ImportError:
            print("WanSeamCC_v2: 'color_matcher' fehlt.")
            return (torch.cat((base_images, overlap_images), dim=0),)

        frames_base = base_images.shape[0]
        frames_new = overlap_images.shape[0]
        actual_overlap = min(overlap_length, frames_base, frames_new)

        if actual_overlap == 0:
            return (torch.cat((base_images, overlap_images), dim=0),)

        # 1. Videos zerlegen (ANDERS HERUM als v1)
        
        # ALTES VIDEO (Base): Wir behalten ALLES.
        # Aber wir trennen den letzten Teil ab, um ihn zu korrigieren.
        part_1_clean = base_images[:-actual_overlap] 
        target_chunk = base_images[-actual_overlap:].clone() # Wird korrigiert (Ende von A)

        # NEUES VIDEO (Overlay): Wir nutzen den Anfang als Referenz, dann werfen wir ihn weg.
        ref_chunk = overlap_images[:actual_overlap]          # Referenz (Anfang von B)
        part_2_clean = overlap_images[actual_overlap:]       # Der Rest von B (bleibt)

        # 2. Color Matching Loop (Ende von A anpassen an Anfang von B)
        cm = ColorMatcher()
        print(f"WanSeamCC_v2: Passe Base-Ende an Overlay-Start an ({actual_overlap} Frames).")

        for i in range(actual_overlap):
            # Ramp Berechnung
            if actual_overlap > 1:
                raw_progress = i / (actual_overlap - 1)
            else:
                raw_progress = 0.0

            if interpolation == "sigmoid":
                progress = raw_progress * raw_progress * (3 - 2 * raw_progress)
            else:
                progress = raw_progress

            # Stärke interpolieren
            current_strength = (1.0 - progress) * match_strength_start + progress * match_strength_end

            if current_strength <= 0.001:
                continue

            # 1:1 Matching
            src_img = target_chunk[i].cpu().numpy()
            ref_img = ref_chunk[i].cpu().numpy()

            try:
                matched_data = cm.transfer(src=src_img, ref=ref_img, method=method)
                matched_tensor = torch.from_numpy(matched_data).to(target_chunk.device)
                
                # Mischen
                target_chunk[i] = torch.lerp(target_chunk[i], matched_tensor, current_strength)
            
            except Exception as e:
                print(f"Fehler Frame {i}: {e}")

        # 3. Alles zusammenfügen
        # [Altes Video Clean] + [Korrigiertes Ende von A] + [Neues Video Rest]
        
        parts = []
        if part_1_clean.shape[0] > 0:
            parts.append(part_1_clean)
        
        parts.append(target_chunk)
        
        if part_2_clean.shape[0] > 0:
            parts.append(part_2_clean)

        result = torch.cat(parts, dim=0)

        return (result,)


NODE_CLASS_MAPPINGS = {
    "RVC_Terminal_Node": RVC_Terminal_Node,
     "Standalone_OverlayTransparentImage": Standalone_OverlayTransparentImage,
    "Standalone_SaveImageClean": Standalone_SaveImageClean,
    "VAEDtypeChecker": VAEDtypeChecker,
    "WanVideoSeamBlender": WanVideoSeamBlender,
    "WanVideoSeamCC": WanVideoSeamCC,
    "WanVideoSeamCC_v2": WanVideoSeamCC_v2,
    
    
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RVC_Terminal_Node": "RVC Terminal (Fixed Paths)",
    "Standalone_OverlayTransparentImage": "Overlay Image (Video Supported)",
    "Standalone_SaveImageClean": "Save Image (No Metadata)",
    "VAEDtypeChecker": "VAE Dtype Checker",
    "WanVideoSeamBlender": "Wan Video Seam Blender",
    "WanVideoSeamCC": "Wan Video Seam (Color Correct & Join)",
    "WanVideoSeamCC_v2": "Wan Video Seam v2 (Correct Base)",
}
