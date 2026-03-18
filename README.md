# 498_Ass3 — Hopfield Network Pattern Recognition Dashboard

A comprehensive Python application for creating, training, and testing binary Hopfield networks with configurable activation functions, learning rules, and Monte Carlo analysis.

## Overview

This project provides an interactive dashboard for:
- **Pattern creation** — design and save binary image patterns of configurable size
- **Noise generation** — create corrupted variants of patterns with controlled noise levels
- **Network training** — train Hopfield networks using multiple learning rules and activation functions
- **Pattern recall** — test trained networks on noisy or original patterns
- **Error reporting** — analyze recall performance with detailed per-image and aggregate metrics
- **Monte Carlo testing** — compare network behavior across different configurations with statistical measures
- **Utility tools** — view patterns, generate bit frequency reports, trim/resize images

## Features

### Core Functionality

**Pattern Management**
- Create and edit binary (0/1) pattern images at custom dimensions
- Save patterns and noisy variants as PNG files
- View pattern galleries and image folders

**Hopfield Network Training**
- **Activation functions:** Sign, Tanh, Softmax
- **Learning rules:** Hebbian, Storkey, Pseudo-Inverse
- **Update modes:** Synchronous (HOPS) and Asynchronous (HOPA)
- Separate trained models for each configuration pair
- Persistent model storage (`.npz` format)

**Pattern Recall**
- Synchronous and asynchronous recall modes
- Capture intermediate HOPA convergence stages
- Visualize recalled patterns and convergence animations
- Energy tracking for both update modes

**Error Analysis**
- Per-image recall error counts and percentages
- Aggregate metrics: Precision, Recall, Specificity, F1-score
- Comparison across repeat runs with mean ± SD
- Bit frequency analysis across pattern sets

**Monte Carlo Analysis (Option 7)**  
- Run configurable number of independent test runs (default 30)
- Generate synthetic noise for each run with adjustable noise percentage
- Compare activation functions OR learning rules with full independence
- Output: Mean error ± 95% confidence interval per test condition
- **Seed policy:** Configurable base seed (blank = random) for non-reproducible runs
- Auto-switches activation-only comparisons to HOPS (HOPA is not meaningful for activation comparison)

### Utilities (Option 8)

1. **Upsize patterns** — Pad images rightward/downward to uniform target dimensions
2. **View folder images** — Display up to 8 images from any folder in a gallery
3. **Create pixelated characters** — Generate 8 clean character patterns (A, B, C, D, P, Q, R, X)
4. **View HOPA animation** — Replay latest asynchronous recall convergence stages
5. **Bit frequency report** — Per-bit frequency statistics and CSV export
6. **Downsize patterns** — Trim trailing all-zero rows/columns uniformly across all images

## Project Structure

```
498_Ass3/
├── dashboard.py              # Interactive main menu
├── hopfield_nn.py            # HN class, training, recall, reporting engines
├── create_img.py             # Pattern creation and editing
├── noise.py                  # Noisy pattern generation  
├── utilities.py              # Pattern viewing, bit analysis, resizing
├── cleanup.py                # File cleanup and temp directory management
├── folder_prefs.py           # User folder preference persistence
├── terminal_out.py           # Logging to terminal_out.txt
│
├── patterns/                 # Training patterns (source reference images)
├── noisy_patterns/           # Test patterns (noise-corrupted variants)
├── patterns_*/               # Output folders from upsize/downsize utilities
├── recall_patterns/          # Recalled output patterns (HOPS, HOPA)
├── temp_patterns/            # HOPA intermediate stage images and animations
│
├── nn_models/                # Trained network persistence
│   ├── HOPS.npz             # Synchronous model + metadata
│   ├── HOPA.npz             # Asynchronous model + metadata
│   ├── LAST_RECALL_SNAPSHOT.npz
│   └── LAST_HOPA_STAGES.npz
│
└── README.md
```

## Requirements

- **Python:** 3.10+
- **Core packages:**
  - `numpy` — numerical computing
  - `matplotlib` — image I/O and visualization
  - `torch` — GPU support detection

### Installation

Create and activate a Python virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install numpy matplotlib torch
```

## Usage

### Start the Dashboard

```powershell
python dashboard.py
```

### Dashboard Menu (Options 0–9)

| Option | Feature | Description |
|--------|---------|-------------|
| **1** | Create/Edit Patterns | Design and save binary patterns to `patterns/` |
| **2** | Generate Noisy Patterns | Create corrupted variants in `noisy_patterns/` with adjustable noise % |
| **3** | Train Networks | Train HOPS and HOPA with chosen activation + learning rule |
| **4** | Recall Patterns | Test both models on a test folder; view results |
| **5** | Recall Error Report | Per-image error counts and metrics from latest recall |
| **6** | Repeat Recall Report | Run 30 (default) recalls on same inputs; report aggregate stats |
| **7** | Monte Carlo Report | Run independent MC trials; compare activations or learning rules |
| **8** | Utilities | Sub-menu: view/upsize/downsize/analyze patterns |
| **9** | Cleanup | Remove generated files and trim logs |
| **0** | Exit | Close application |

### Typical Workflow

#### Example 1: Train and Test a Network

1. **Option 1:** Create patterns (e.g., 10 patterns of 12×10 size)
2. **Option 3:** Train with `Sign` activation + `Hebbian` learning
3. **Option 2:** Generate noisy patterns at 20% corruption
4. **Option 4:** Recall from `noisy_patterns/` folder
5. **Option 5:** View per-image recall errors
6. **Option 6:** Run 30 repeat recalls for mean ± SD

#### Example 2: Compare Activation Functions with Monte Carlo

1. **Option 3:** Train one network (store baseline)
2. **Option 7:**
   - Choose **Compare Activation** mode
   - Pick **Learning mode:** Hebbian (fixed)
   - Pick **Activations:** All (Sign, Tanh, Softmax)
   - Recall mode: **HOPS** (auto-suggested; HOPA not valid for activation comparison)
   - Runs: 30, Noise: 20%
   - **Base seed:** Leave blank for random, or enter integer for reproducible seed
3. Output: Table with mean error ± 95% CI for each activation

#### Example 3: Downsize Padded Patterns

1. **Option 8 → 6:** Downsize patterns utility
2. Select source folder (e.g., `patterns_120_100`)
3. Utility scans all images, finds minimum trailing zero rows/columns
4. Trims ALL images uniformly → output to `patterns_trimmed/` with same dimensions

## Configuration

### Network Parameters

**Fixed at startup:**
- `GRID_ROWS`, `GRID_COLS` — image dimension (defaults: 12, 10)
- Model paths — `nn_models/HOPS.npz`, `nn_models/HOPA.npz`
- Recall folder — `noisy_patterns/` (user-selectable in menu)

**User-selectable per run:**
- **Activation:** Sign (default), Tanh, Softmax
- **Learning rule:** Hebbian (default), Storkey, Pseudo-Inverse
- **Recall mode:** HOPS (synchronous), HOPA (asynchronous)
- **Noise level:** 0–100% bit-flip rate
- **Repeat/MC runs:** Any positive integer (defaults: 10 for repeat, 30 for MC)
- **Base seed** (MC only): Random (blank) or fixed integer

### Activation Functions

| Function  | Definition | Behavior |
|-----------|-----------|----------|
| **Sign** | `sign(x)` | Threshold at 0: `x ≥ 0 → +1`, else `−1` |
| **Tanh** | `tanh(x)` projected | Smooth curve, then thresholded like Sign |
| **Softmax** | Winner-take-all | Probabilistic activation; highest logit wins |

**Note:** SIGN and TANH are mathematically equivalent in this implementation (both threshold at 0 for bipolar output).

### Learning Rules

| Rule | Training Complexity | Best For |
|------|---------------------|----------|
| **Hebbian** | O(N×P) — fast | Quick training, simpler patterns |
| **Storkey** | O(N²×P) — slower | Better pattern overlap handling |
| **Pseudo-Inverse** | O(N² × M) — invertible if well-conditioned | Exact recall of training set when possible |

## Output Files

### Trained Models

- `nn_models/HOPS.npz` — weights + metadata (activation, learning rule, grid shape)
- `nn_models/HOPA.npz` — asynchronous weights + metadata

### Recall Artifacts

- `recall_patterns/HOPS_pattern_X.img` — recalled patterns (synchronous)
- `recall_patterns/HOPA_pattern_X.img` — recalled patterns (asynchronous)
- `temp_patterns/run_YYYYMMDD_HHMMSS/` — HOPA intermediate convergence stages

### Analysis Reports

- `patterns/bit_frequency_report.csv` — per-bit frequency analysis
- `terminal_out.txt` — full terminal log (overwritten each run)

## Notes and Tips

### Seeding Behavior

- **Repeat Recall (Option 6):** Uses SAME noisy inputs for all repeats (intended for noise stability)
- **Monte Carlo (Option 7):** Uses DIFFERENT noise each run (independent trials)
  - Default base seed: **42** (reproducible)
  - Blank base seed: **random** (non-reproducible, recommended for experimentation)

### Common Issues

**Identical errors across activation comparisons?**
- TANH and SIGN are equivalent in bipolar projection — choose different pairs (e.g., SIGN vs SOFTMAX)
- HOPA activation comparison is not meaningful; tool auto-switches to HOPS

**Recall error unchanged?**
- Check that noisy patterns are actually different from originals
- Verify model was trained on patterns matching the test image dimensions

**Memory issues with large patterns?**
- Reduce pattern count or image size; Storkey/Pseudo-Inverse scale as O(N²)

## Development Notes

- Folder preferences persist in `.folder_prefs.json`
- All binary patterns use convention: `0 = white`, `1 = black`
- Patterns must all be same size within a folder for training/analysis
- CUDA GPU support is auto-detected; falls back to CPU

## License

Educational/research project. Use freely.
