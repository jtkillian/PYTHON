# Echo — Usage Guide

## Installation

Install the GPU-first stack (falls back to CPU automatically):

```bash
uv pip install --extra-index-url https://download.pytorch.org/whl/cu124 torch torchaudio numpy scipy librosa soundfile matplotlib imageio imageio-ffmpeg tqdm rich typer cupy-cuda12x nvidia-cuda-runtime-cu12 nvidia-cublas-cu12 nvidia-cudnn-cu12 sounddevice
```

Copy or record WAV files into `assets/wav/`. Run `python scripts/generate_sample_wav.py` to materialise the bundled `sine.wav`
tone for quick testing.

## Launching the app

| Command | Purpose |
| --- | --- |
| `python spectro3d.py demo assets/wav/sine.wav --mp4 dist/videos/sine.mp4 --fps 30` | Precompute, open the UI, and export an MP4 during the session. |
| `python spectro3d.py view assets/wav/song.wav` | Precompute (cache) then open the interactive viewer. |
| `python spectro3d.py render-mp4 assets/wav/song.wav --out dist/videos/song.mp4 --fps 60` | Render a video only (no window). |
| `python spectro3d.py play assets/wav/song.wav` | Start playback with the 4-pane UI. |

All commands reuse cached STFT data from `dist/caches/` when the audio file and configuration match.

## The four-pane layout

The app opens one Matplotlib window with a `2x2` grid:

1. **Top-left — 3D spectrogram surface:** displays the gated time slice of the spectrogram. Rotate/zoom with the mouse. Toggle fidelity with `P`.
2. **Top-right — 2D heatmap:** full spectrogram with a vertical cursor following the current time.
3. **Bottom-left — Waveform:** click anywhere to jump playback and cursor.
4. **Bottom-right — Spectrum slice:** shows the magnitude spectrum at the current time index.

A slider below the panes provides continuous scrubbing.

## Scrubbing and controls

- **Left / Right arrows:** jump backward/forward 5 seconds.
- **Comma / Period (, / .):** jump backward/forward by the fine step (1.0 s or 0.5 s).
- **Click "fine=..." in the HUD:** toggle the fine step between 1.0 and 0.5 seconds.
- **[ / ]:** decrease/increase the gate width (2–60 s).
- **Q / E:** decrease/increase the gate scroll rate (0.25×–4×).
- **P:** toggle Performance ↔ Hi-Fi fidelity. Performance mode subsamples the 3D mesh for smoother interactivity.
- **Click in heatmap or waveform:** seek to that time.
- **Slider:** scrub continuously.

The on-screen HUD in the lower-right shows the active key bindings and live values for the gate width, gate rate, fidelity mode, and fine step.

## Adjusting analysis parameters

Edit `spectro3d.py` or pass CLI overrides to tweak the analysis. For example:

```bash
python spectro3d.py view assets/wav/song.wav --n-fft 4096 --hop-length 512 --db-floor -100
```

Available options include the FFT size, hop length, window type, decibel floor/ceiling, mono fold-down, gate defaults, and more. All options participate in the cache hash, so the STFT recomputes automatically when you change them.

## Troubleshooting

- **No CUDA GPU:** The log will note the CPU path; performance may be slower but the UI remains functional.
- **Audio device errors:** Ensure a playback device is available or run with `--no-playback` (see CLI help). The UI still works without audio.
- **Long MP4 renders:** Add `--fast` to reuse the performance LoD in the 3D pane and render only the current gate.
- **Cache growth:** Delete files under `dist/caches/` to regenerate from scratch.
