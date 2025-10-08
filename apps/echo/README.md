# Echo 3D Spectrogram

Echo is a GPU-friendly interactive 3D spectrogram viewer and renderer. It prerenders Short Time Fourier Transform (STFT) data with CUDA when available, caches the result to disk, and serves a single-window Matplotlib UI with four coordinated panes for deep audio inspection.

## Quick start

1. Install dependencies (CUDA builds are preferred automatically when available):

   ```bash
   uv pip install --extra-index-url https://download.pytorch.org/whl/cu124 torch torchaudio numpy scipy librosa soundfile matplotlib imageio imageio-ffmpeg tqdm rich typer cupy-cuda12x nvidia-cuda-runtime-cu12 nvidia-cublas-cu12 nvidia-cudnn-cu12 sounddevice
   ```

2. Drop WAV files in `assets/wav/` (for example `assets/wav/example.wav`).

3. Run one of the CLI commands below from this directory:

   ```bash
   python spectro3d.py demo assets/wav/example.wav --mp4 dist/videos/example.mp4 --fps 30
   python spectro3d.py view assets/wav/song.wav
   python spectro3d.py render-mp4 assets/wav/song.wav --out dist/videos/song.mp4 --fps 60
   ```

The app stores prerendered STFT caches inside `dist/caches/` keyed by the audio identity and relevant configuration hash. MP4 exports land in `dist/videos/`.

On supported NVIDIA GPUs, the STFT stage uses PyTorch CUDA kernels. When CUDA is not available the pipeline automatically falls back to the CPU (librosa), trading speed for portability.

## Key bindings

| Action | Binding |
| --- | --- |
| Jump ±5 seconds | ← / → |
| Fine scrub ±{fine_step}s | , / . |
| Adjust gate width | [ / ] |
| Adjust gate scroll rate | Q / E |
| Toggle performance fidelity | P |
| Click waveform or spectrogram | Seek to clicked time |
| Slider | Continuous scrub |

The on-screen HUD in the lower-right corner always shows the latest values for the gate width, scroll rate, fidelity mode, and current fine scrub step.
