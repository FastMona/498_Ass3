# 498_Ass3 - Hopfield Pattern Dashboard

A small Python project for creating binary image patterns, generating noisy variants, training Hopfield models, recalling patterns, and reporting recall error.

## Features

- Create and edit `12x10` binary pattern images (`0 = white`, `1 = black`)
- Save and preview recent patterns
- Generate noisy pattern sets from existing pattern images
- Train two Hopfield recall modes:
  - `HOPS` (synchronous update)
  - `HOPA` (asynchronous update)
- Recall patterns from a test folder and view outputs
- Report per-image recall error against original patterns
- Repeat recall report with aggregate `#Errors`, `P`, `R`, `S`, and `F` metrics
- Optional cleanup utility for generated files

## Project Structure

- `dashboard.py` - main interactive menu
- `create_img.py` - create/edit/save pattern PNGs
- `noise.py` - generate noisy pattern PNGs
- `hopfield_nn.py` - train/recall/report logic and model persistence
- `cleanup.py` - remove generated PNG/temp files and trim terminal log
- `terminal_out.py` - tee stdout/stderr to `terminal_out.txt`
- `patterns/` - source pattern images
- `noisy_patterns/` - generated noisy images
- `nn_models/` - saved model files and recall snapshots

## Requirements

- Python 3.10+
- Packages:
  - `numpy`
  - `matplotlib`
  - `torch`

Install dependencies:

```powershell
pip install numpy matplotlib torch
```

## Run

Start the dashboard:

```powershell
python dashboard.py
```

You can also run modules directly:

```powershell
python create_img.py
python noise.py
python hopfield_nn.py
python cleanup.py
```

## Typical Workflow

1. Run `python dashboard.py`
2. Option `1`: Create/save patterns in `patterns/`
3. Option `2`: Generate noisy patterns in `noisy_patterns/`
4. Option `3`: Train Hopfield models (`HOPS.npz`, `HOPA.npz`)
5. Option `4`: Recall and visualize outputs
6. Option `5`: Print recall error report
7. Option `6`: Run repeat recall report (choose repeat count)

## Notes

- Pattern dimensions are fixed at `12x10` (`GRID_ROWS = 12`, `GRID_COLS = 10`).
- Training and recall expect PNG files that match this exact size.
- `terminal_out.txt` is overwritten each run when logging is enabled.
