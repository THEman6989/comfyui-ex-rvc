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

NODE_CLASS_MAPPINGS = {
    "RVC_Terminal_Node": RVC_Terminal_Node
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RVC_Terminal_Node": "RVC Terminal (Fixed Paths)"
}
