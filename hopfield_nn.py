from math import sqrt
from pathlib import Path
from typing import cast
from datetime import datetime
import shutil
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from multiprocessing import get_context

from create_img import GRID_ROWS, GRID_COLS, ensure_patterns_dir, load_pattern_image
from folder_prefs import prompt_for_folder
from terminal_out import install_terminal_output_logger


MODELS_DIR = Path(__file__).resolve().parent / "nn_models"
HOPS_MODEL_PATH = MODELS_DIR / "HOPS.npz"
HOPA_MODEL_PATH = MODELS_DIR / "HOPA.npz"
LAST_RECALL_SNAPSHOT_PATH = MODELS_DIR / "LAST_RECALL_SNAPSHOT.npz"
NEURON_COUNT = GRID_ROWS * GRID_COLS
DEFAULT_ACTIVATION = "sign"
DEFAULT_LEARNING_MODE = "hebbian"
DEFAULT_RECALL_TEST_FOLDER = "noisy_patterns"
RECALL_PATTERNS_DIR = Path(__file__).resolve().parent / "recall_patterns"
TEMP_PATTERNS_DIR = Path(__file__).resolve().parent / "temp_patterns"
LAST_HOPA_STAGES_PATH = MODELS_DIR / "LAST_HOPA_STAGES.npz"


def _set_window_title(fig, window_title: str | None) -> None:
    """Set matplotlib window title when backend supports it."""
    manager = getattr(fig.canvas, "manager", None)
    if manager is None or not window_title:
        return
    try:
        manager.set_window_title(window_title)
    except Exception:
        return


def _show_gallery_window_process(
    images: list[np.ndarray],
    image_titles: list[str],
    suptitle: str,
    window_title: str | None,
    use_binary_colormap: bool,
) -> None:
    """Render up to 8 images in a 2x4 gallery in a child process."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    _set_window_title(fig, window_title)
    axes_flat = axes.flatten()
    cmap = ListedColormap(["white", "black"])

    for index, axis in enumerate(axes_flat):
        if index >= len(images):
            axis.axis("off")
            continue

        image = images[index]
        if use_binary_colormap:
            axis.imshow(image, cmap=cmap, vmin=0, vmax=1)
        else:
            if image.ndim == 3:
                axis.imshow(image)
            else:
                axis.imshow(image, cmap=cmap, vmin=0, vmax=1)
        axis.set_title(image_titles[index])
        axis.set_xticks([])
        axis.set_yticks([])

    fig.suptitle(suptitle)
    plt.tight_layout()
    plt.show()


def show_gallery_window(
    images: list[np.ndarray],
    image_titles: list[str],
    suptitle: str,
    window_title: str | None = None,
    use_binary_colormap: bool = False,
) -> None:
    """Start a process that shows a gallery window."""
    context = get_context("spawn")
    process = context.Process(
        target=_show_gallery_window_process,
        args=(
            [np.asarray(image) for image in images],
            image_titles,
            suptitle,
            window_title,
            use_binary_colormap,
        ),
        daemon=True,
    )
    process.start()


class HopfieldNetwork:
    """Simple Hopfield network for bipolar patterns (-1, +1)."""

    def __init__(self, size: int, activation: str = DEFAULT_ACTIVATION) -> None:
        self.size = size
        self.activation = activation
        self.weights = np.zeros((size, size), dtype=float)

    def apply_activation(self, values: np.ndarray) -> np.ndarray:
        """Apply configured activation and return bipolar states (-1/+1)."""
        if self.activation in {"tanh", "tan"}:
            activated = np.tanh(values)
            return np.where(activated >= 0, 1, -1)
        if self.activation == "softmax":
            # Winner-take-all bipolar projection from a softmax distribution.
            if values.size <= 1:
                return np.where(values >= 0, 1, -1)
            shifted = values - np.max(values)
            exp_values = np.exp(shifted)
            denom = np.sum(exp_values)
            if denom <= 0 or not np.isfinite(denom):
                return np.where(values >= 0, 1, -1)
            probabilities = exp_values / denom
            winner_index = int(np.argmax(probabilities))
            activated = np.full(values.shape, -1, dtype=int)
            activated[winner_index] = 1
            return activated
        if self.activation in {"sign", "sin"}:
            # Keep legacy "sin" model files compatible by mapping them to sign.
            return np.where(values >= 0, 1, -1)
        return np.where(values >= 0, 1, -1)

    def train_hebbian(self, patterns: list[np.ndarray], label: str = "HEBB") -> None:
        """Train weights using batch Hebbian learning."""
        self.weights.fill(0)
        for index, pattern in enumerate(patterns, start=1):
            self.weights += np.outer(pattern, pattern)
            print(f"[{label}] Hebbian processed pattern {index}/{len(patterns)}")

        self.weights /= self.size
        np.fill_diagonal(self.weights, 0)
        print(f"[{label}] Hebbian training complete.")

    def train_storkey(self, patterns: list[np.ndarray], label: str = "STORK") -> None:
        """Train weights using the Storkey learning rule."""
        self.weights.fill(0)
        n = self.size

        for pattern_index, pattern in enumerate(patterns, start=1):
            for i in range(n):
                for j in range(i + 1, n):
                    h_i = np.dot(self.weights[i, :], pattern) - self.weights[i, i] * pattern[i] - self.weights[i, j] * pattern[j]
                    h_j = np.dot(self.weights[j, :], pattern) - self.weights[j, j] * pattern[j] - self.weights[j, i] * pattern[i]
                    delta = (pattern[i] * pattern[j] - pattern[i] * h_j - h_i * pattern[j]) / n
                    self.weights[i, j] += delta
                    self.weights[j, i] += delta

            np.fill_diagonal(self.weights, 0)
            print(f"[{label}] Storkey processed pattern {pattern_index}/{len(patterns)}")

        print(f"[{label}] Storkey training complete.")

    def train_pseudo_inverse(self, patterns: list[np.ndarray], label: str = "PINV") -> None:
        """Train weights using the pseudo-inverse learning rule."""
        if not patterns:
            self.weights.fill(0)
            return

        x = np.vstack(patterns)
        print(f"[{label}] Pseudo-inverse matrix build for {x.shape[0]} pattern(s)...")

        gram = x @ x.T
        gram_inv = np.linalg.pinv(gram)
        self.weights = x.T @ gram_inv @ x
        np.fill_diagonal(self.weights, 0)
        print(f"[{label}] Pseudo-inverse training complete.")

    def energy(self, state: np.ndarray) -> float:
        """Return Hopfield energy for a bipolar state."""
        return float(-0.5 * state.T @ self.weights @ state)

    def recall_synchronous(self, pattern: np.ndarray, steps: int = 1) -> np.ndarray:
        """Recall using synchronous state updates."""
        state = pattern.copy()
        for _ in range(steps):
            activation = self.weights @ state
            state = self.apply_activation(activation)
        return state

    def recall_asynchronous(
        self,
        pattern: np.ndarray,
        steps: int = 1,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """Recall using asynchronous state updates."""
        state = pattern.copy()
        if rng is None:
            rng = np.random.default_rng()
        for _ in range(steps):
            order = rng.permutation(self.size)
            for neuron_index in order:
                activation = np.dot(self.weights[neuron_index], state)
                state[neuron_index] = int(self.apply_activation(np.array([activation]))[0])
        return state


def grid_to_bipolar_vector(grid: list[list[int]]) -> np.ndarray:
    """Convert binary grid (0/1) to bipolar vector (-1/+1)."""
    array = np.asarray(grid, dtype=int)
    return np.where(array.reshape(-1) == 1, 1, -1)


def load_binary_png_any_size(image_path: Path) -> np.ndarray | None:
    """Load a PNG as a 2D binary grid (0/1) at native dimensions."""
    try:
        data = plt.imread(image_path)
    except OSError:
        return None

    if data.ndim == 3:
        data = data[..., :3].mean(axis=2)

    if data.ndim != 2:
        return None

    return (np.asarray(data) < 0.5).astype(np.uint8)


def load_training_patterns(folder: Path) -> tuple[list[np.ndarray], list[Path], tuple[int, int] | None]:
    """Load valid training patterns from PNG files in a folder."""
    image_files = sorted(folder.glob("*.png"))
    vectors: list[np.ndarray] = []
    used_files: list[Path] = []
    expected_shape: tuple[int, int] | None = None

    if not image_files:
        return vectors, used_files, expected_shape

    print(f"Found {len(image_files)} PNG file(s). Validating consistent dimensions...")
    for index, image_file in enumerate(image_files, start=1):
        grid_array = load_binary_png_any_size(image_file)
        if grid_array is None:
            print(f"Skipping {image_file.name}: unreadable or non-2D PNG.")
            continue

        current_shape = (int(grid_array.shape[0]), int(grid_array.shape[1]))
        if expected_shape is None:
            expected_shape = current_shape
            print(f"Detected training pattern size: {expected_shape[0]}x{expected_shape[1]}")
        elif current_shape != expected_shape:
            print(
                f"Skipping {image_file.name}: size {current_shape[0]}x{current_shape[1]} does not match "
                f"expected {expected_shape[0]}x{expected_shape[1]}."
            )
            continue

        vectors.append(np.where(grid_array.reshape(-1) == 1, 1, -1))
        used_files.append(image_file)
        print(f"Loaded pattern {len(vectors)} from {image_file.name} ({index}/{len(image_files)})")

    return vectors, used_files, expected_shape


def read_activation_choice() -> str:
    """Prompt user for activation choice."""
    while True:
        print("Activation function:")
        print("1. Sign")
        print("2. Tanh")
        print("3. SoftMax")
        value = input("Choose activation [1]: ").strip().lower()
        if value == "":
            return "sign"
        if value == "1" or value in {"sign", "sin"}:
            return "sign"
        if value == "2" or value in {"tan", "tanh"}:
            return "tanh"
        if value == "3" or value in {"softmax", "soft_max", "soft"}:
            return "softmax"
        print("Invalid activation. Enter 1 (Sign), 2 (Tanh), or 3 (SoftMax).")


def read_learning_mode_choice() -> str:
    """Prompt user for learning mode."""
    while True:
        print("Learning mode:")
        print("1. Hebbian")
        print("2. Storkey")
        print("3. Pseudo Inverse")
        value = input("Choose learning mode [1]: ").strip().lower()
        if value == "":
            return "hebbian"
        if value == "1" or value in {"hebbian", "hebb"}:
            return "hebbian"
        if value == "2" or value in {"storkey", "stork"}:
            return "storkey"
        if value == "3" or value in {"pseudo_inv", "pseudo-inverse", "pseudoinverse", "pinv"}:
            return "pseudo_inv"
        print("Invalid learning mode. Enter 1 (Hebbian), 2 (Storkey), or 3 (Pseudo Inverse).")


def ensure_models_dir() -> Path:
    """Ensure model output folder exists and return its path."""
    MODELS_DIR.mkdir(exist_ok=True)
    return MODELS_DIR


def ensure_recall_patterns_dir() -> Path:
    """Ensure recalled patterns output folder exists and return its path."""
    RECALL_PATTERNS_DIR.mkdir(exist_ok=True)
    return RECALL_PATTERNS_DIR


def ensure_temp_patterns_dir() -> Path:
    """Ensure temporary intermediate-stage output folder exists."""
    TEMP_PATTERNS_DIR.mkdir(exist_ok=True)
    return TEMP_PATTERNS_DIR


def reset_temp_patterns_dir() -> Path:
    """Clear all prior intermediate-stage output and recreate temp folder."""
    if TEMP_PATTERNS_DIR.exists():
        shutil.rmtree(TEMP_PATTERNS_DIR, ignore_errors=True)
    TEMP_PATTERNS_DIR.mkdir(exist_ok=True)
    return TEMP_PATTERNS_DIR


def save_network_params(
    model_path: Path,
    model_name: str,
    network: HopfieldNetwork,
    training_folder: Path,
    pattern_count: int,
    learning_mode: str,
    grid_shape: tuple[int, int],
) -> None:
    """Persist network parameters so retraining is not required on next startup."""
    ensure_models_dir()
    np.savez(
        model_path,
        weights=network.weights,
        activation=network.activation,
        size=network.size,
        grid_rows=grid_shape[0],
        grid_cols=grid_shape[1],
        training_folder=str(training_folder),
        pattern_count=pattern_count,
        learning_mode=learning_mode,
        model_name=model_name,
    )
    print(f"[{model_name}] Parameters saved: {model_path}")


def train_network(network: HopfieldNetwork, patterns: list[np.ndarray], learning_mode: str, label: str) -> None:
    """Train one network with the selected learning rule."""
    if learning_mode == "hebbian":
        network.train_hebbian(patterns, label=label)
        return
    if learning_mode == "storkey":
        network.train_storkey(patterns, label=label)
        return
    network.train_pseudo_inverse(patterns, label=label)


def track_energy_synchronous(network: HopfieldNetwork, pattern: np.ndarray, steps: int = 5) -> list[float]:
    """Track energy across synchronous recall updates."""
    state = pattern.copy()
    energies = [network.energy(state)]
    for _ in range(steps):
        state = network.recall_synchronous(state, steps=1)
        energies.append(network.energy(state))
    return energies


def track_energy_asynchronous(network: HopfieldNetwork, pattern: np.ndarray, steps: int = 5) -> list[float]:
    """Track energy across asynchronous recall updates."""
    state = pattern.copy()
    energies = [network.energy(state)]
    for _ in range(steps):
        state = network.recall_asynchronous(state, steps=1)
        energies.append(network.energy(state))
    return energies


def get_trained_model_labels() -> list[str]:
    """Return trained model labels available on disk."""
    trained: list[str] = []

    if HOPS_MODEL_PATH.exists():
        trained.append("HOPS")

    if HOPA_MODEL_PATH.exists():
        trained.append("HOPA")

    return trained


def bipolar_vector_to_grid(vector: np.ndarray, grid_shape: tuple[int, int]) -> np.ndarray:
    """Convert bipolar vector (-1/+1) back to binary image grid (0/1)."""
    binary = np.where(vector.reshape(-1) == 1, 1, 0)
    return binary.reshape(grid_shape)


def load_network_from_file(model_path: Path) -> HopfieldNetwork | None:
    """Load a trained network from saved npz parameters."""
    if not model_path.exists():
        return None

    try:
        data = np.load(model_path, allow_pickle=True)
    except OSError:
        return None

    try:
        size = int(data["size"])
        activation = str(data["activation"])
        weights = np.asarray(data["weights"], dtype=float)
    except KeyError:
        return None

    # Normalize legacy activation labels from older model files.
    if activation == "tan":
        activation = "tanh"
    elif activation == "sin":
        activation = "sign"

    if weights.shape != (size, size):
        return None

    network = HopfieldNetwork(size=size, activation=activation)
    network.weights = weights
    return network


def load_model_metadata(model_path: Path) -> dict[str, str] | None:
    """Load display metadata for a saved model file."""
    if not model_path.exists():
        return None

    try:
        data = np.load(model_path, allow_pickle=True)
    except OSError:
        return None

    try:
        size = int(data["size"])
        activation = str(data["activation"])
    except KeyError:
        return None

    # Normalize legacy activation labels for user-facing metadata.
    if activation == "tan":
        activation = "tanh"
    elif activation == "sin":
        activation = "sign"

    learning_mode = str(data["learning_mode"]) if "learning_mode" in data else "unknown"
    pattern_count = str(int(data["pattern_count"])) if "pattern_count" in data else "unknown"
    training_folder = str(data["training_folder"]) if "training_folder" in data else "unknown"
    if "grid_rows" in data and "grid_cols" in data:
        grid_rows = str(int(data["grid_rows"]))
        grid_cols = str(int(data["grid_cols"]))
    else:
        grid_rows = str(GRID_ROWS)
        grid_cols = str(GRID_COLS)

    return {
        "size": str(size),
        "activation": activation,
        "learning_mode": learning_mode,
        "pattern_count": pattern_count,
        "training_folder": training_folder,
        "grid_rows": grid_rows,
        "grid_cols": grid_cols,
    }


def get_model_grid_shape(model_path: Path, network: HopfieldNetwork) -> tuple[int, int] | None:
    """Resolve model grid shape from metadata and validate it against neuron count."""
    metadata = load_model_metadata(model_path)
    if metadata is None:
        return None

    try:
        rows = int(metadata["grid_rows"])
        cols = int(metadata["grid_cols"])
    except (KeyError, ValueError):
        return None

    if rows <= 0 or cols <= 0:
        return None
    if rows * cols != network.size:
        return None

    return rows, cols


def infer_model_grid_shape_from_files(network: HopfieldNetwork, image_files: list[Path]) -> tuple[int, int] | None:
    """Infer model grid shape from candidate PNG files using neuron count compatibility."""
    matched_shapes: list[tuple[int, int]] = []
    for image_path in image_files:
        grid_array = load_binary_png_any_size(image_path)
        if grid_array is None:
            continue

        rows, cols = int(grid_array.shape[0]), int(grid_array.shape[1])
        if rows * cols != network.size:
            continue

        current_shape = (rows, cols)
        if current_shape not in matched_shapes:
            matched_shapes.append(current_shape)

    if len(matched_shapes) == 1:
        return matched_shapes[0]
    return None


def resolve_model_grid_shape(model_path: Path, network: HopfieldNetwork, image_files: list[Path]) -> tuple[int, int] | None:
    """Resolve model shape from metadata or infer from candidate files for legacy compatibility."""
    metadata_shape = get_model_grid_shape(model_path, network)
    if metadata_shape is not None:
        return metadata_shape
    return infer_model_grid_shape_from_files(network, image_files)


def save_recent_recall_snapshot(
    test_folder: Path,
    valid_files: list[Path],
    hops_recalled_grids: list[np.ndarray],
    hopa_recalled_grids: list[np.ndarray],
    grid_shape: tuple[int, int],
) -> None:
    """Persist latest recall outputs for downstream Option 5 analysis."""
    ensure_models_dir()

    if not valid_files:
        return

    file_names = np.asarray([file_path.name for file_path in valid_files], dtype=str)
    hops_array = np.stack([np.asarray(grid, dtype=np.uint8) for grid in hops_recalled_grids], axis=0)
    hopa_array = np.stack([np.asarray(grid, dtype=np.uint8) for grid in hopa_recalled_grids], axis=0)

    np.savez(
        LAST_RECALL_SNAPSHOT_PATH,
        test_folder=str(test_folder),
        file_names=file_names,
        hops_recalled=hops_array,
        hopa_recalled=hopa_array,
        grid_rows=grid_shape[0],
        grid_cols=grid_shape[1],
    )


def load_recent_recall_snapshot() -> dict[str, object] | None:
    """Load latest persisted recall outputs for Option 5 analysis."""
    if not LAST_RECALL_SNAPSHOT_PATH.exists():
        return None

    try:
        data = np.load(LAST_RECALL_SNAPSHOT_PATH, allow_pickle=True)
    except OSError:
        return None

    required_keys = {"test_folder", "file_names", "hops_recalled", "hopa_recalled"}
    if not required_keys.issubset(set(data.files)):
        return None

    file_names = [str(name) for name in np.asarray(data["file_names"]).tolist()]
    hops_recalled = np.asarray(data["hops_recalled"], dtype=np.uint8)
    hopa_recalled = np.asarray(data["hopa_recalled"], dtype=np.uint8)

    if hops_recalled.ndim != 3 or hopa_recalled.ndim != 3:
        return None
    if hops_recalled.shape != hopa_recalled.shape:
        return None
    if hops_recalled.shape[0] != len(file_names):
        return None

    if "grid_rows" in data and "grid_cols" in data:
        grid_rows = int(data["grid_rows"])
        grid_cols = int(data["grid_cols"])
    else:
        grid_rows = int(hops_recalled.shape[1])
        grid_cols = int(hops_recalled.shape[2])

    if grid_rows <= 0 or grid_cols <= 0:
        return None
    if hops_recalled.shape[1:] != (grid_rows, grid_cols):
        return None

    return {
        "test_folder": str(data["test_folder"]),
        "file_names": file_names,
        "hops_recalled": hops_recalled,
        "hopa_recalled": hopa_recalled,
        "grid_rows": grid_rows,
        "grid_cols": grid_cols,
    }


def display_recalled_patterns(test_files: list[Path], recalled_grids: list[np.ndarray], model_name: str) -> None:
    """Display recalled images in a 2x4 figure."""
    image_titles = [test_file.stem for test_file in test_files]
    images = [np.asarray(grid, dtype=np.uint8) for grid in recalled_grids]

    ordered_items = sorted(zip(image_titles, images), key=lambda item: item[0].lower())
    image_titles = [item[0] for item in ordered_items]
    images = [item[1] for item in ordered_items]

    show_gallery_window(
        images,
        image_titles,
        suptitle=f"{model_name} Recalled Patterns (Top 8)",
        window_title=f"{model_name} Recalled Patterns",
        use_binary_colormap=True,
    )


def save_recalled_patterns(test_files: list[Path], recalled_grids: list[np.ndarray], model_name: str) -> int:
    """Save recalled images to recall_patterns using MODEL_patternXX.img naming."""
    output_folder = ensure_recall_patterns_dir()
    saved_count = 0
    cmap = ListedColormap(["white", "black"])

    for test_file, recalled_grid in zip(test_files, recalled_grids):
        pattern_stem = infer_reference_pattern_stem(test_file.stem)
        output_name = f"{model_name}_{pattern_stem}.img"
        output_path = output_folder / output_name
        grid_array = np.asarray(recalled_grid, dtype=np.uint8)
        # Force PNG encoding while honoring requested .img file naming.
        plt.imsave(output_path, grid_array, cmap=cmap, vmin=0, vmax=1, format="png")
        saved_count += 1

    return saved_count


def recall_asynchronous_pixel_stages_until_stable(
    network: HopfieldNetwork,
    pattern: np.ndarray,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    """Capture HOPA states after each pixel update until a full sweep makes no changes."""
    state = pattern.copy()
    stages = [state.copy()]

    while True:
        changed_this_sweep = False
        order = rng.permutation(network.size)
        for neuron_index in order:
            activation = np.dot(network.weights[neuron_index], state)
            new_value = int(network.apply_activation(np.array([activation]))[0])
            if int(state[neuron_index]) != new_value:
                state[neuron_index] = new_value
                stages.append(state.copy())
                changed_this_sweep = True

        if not changed_this_sweep:
            break

    return stages


def _save_hopa_stage_images_for_pattern(run_folder: Path, pattern_stem: str, stage_grids: np.ndarray) -> int:
    """Save stage grids for one pattern as ordered PNG frames."""
    pattern_folder = run_folder / pattern_stem
    pattern_folder.mkdir(exist_ok=True)
    cmap = ListedColormap(["white", "black"])

    for stage_index, stage_grid in enumerate(stage_grids):
        if stage_index == 0:
            stage_name = f"step_{stage_index:03d}_noisy.png"
        elif stage_index == int(stage_grids.shape[0]) - 1:
            stage_name = f"step_{stage_index:03d}_recalled.png"
        else:
            stage_name = f"step_{stage_index:03d}.png"

        output_path = pattern_folder / stage_name
        plt.imsave(output_path, stage_grid, cmap=cmap, vmin=0, vmax=1)

    return int(stage_grids.shape[0])


def save_hopa_intermediate_stages(
    test_folder: Path,
    valid_files: list[Path],
    hopa_stage_grids: list[np.ndarray],
    grid_shape: tuple[int, int],
) -> Path | None:
    """Persist latest HOPA intermediate recall stages to temp folder and snapshot npz."""
    if not valid_files or not hopa_stage_grids:
        return None

    ensure_models_dir()
    reset_temp_patterns_dir()
    run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_folder = ensure_temp_patterns_dir() / run_name
    run_folder.mkdir(exist_ok=True)

    stage_arrays = [np.asarray(stages, dtype=np.uint8) for stages in hopa_stage_grids]
    frame_counts = np.asarray([int(stage_array.shape[0]) for stage_array in stage_arrays], dtype=int)
    stage_objects = np.empty(len(stage_arrays), dtype=object)
    for index, stage_array in enumerate(stage_arrays):
        stage_objects[index] = stage_array

    file_names = np.asarray([file_path.name for file_path in valid_files], dtype=str)

    for file_path, stage_grids in zip(valid_files, stage_arrays):
        _save_hopa_stage_images_for_pattern(run_folder, file_path.stem, stage_grids)

    np.savez(
        LAST_HOPA_STAGES_PATH,
        run_folder=str(run_folder),
        test_folder=str(test_folder),
        file_names=file_names,
        stages=stage_objects,
        frame_counts=frame_counts,
        grid_rows=grid_shape[0],
        grid_cols=grid_shape[1],
    )

    return run_folder


def run_pattern_recall() -> None:
    """Recall up to 8 patterns from a test folder using both trained models."""
    print("\n=== Pattern Recall ===")
    default_test_folder = Path(__file__).resolve().parent / DEFAULT_RECALL_TEST_FOLDER
    test_folder = prompt_for_folder("hopfield.test_folder", "Folder to test", default_test_folder)

    if not test_folder.exists() or not test_folder.is_dir():
        print(f"Folder not found: {test_folder}")
        return

    hops_network = load_network_from_file(HOPS_MODEL_PATH)
    if hops_network is None:
        print("Trained HOPS model not found or invalid. Run option 3 first.")
        return

    hopa_network = load_network_from_file(HOPA_MODEL_PATH)
    if hopa_network is None:
        print("Trained HOPA model not found or invalid. Run option 3 first.")
        return

    test_files = sorted(
        test_folder.glob("*.png"),
        key=lambda file_path: file_path.stat().st_mtime,
        reverse=True,
    )[:8]

    if not test_files:
        print(f"No PNG images found in: {test_folder}")
        return

    hops_shape = resolve_model_grid_shape(HOPS_MODEL_PATH, hops_network, test_files)
    hopa_shape = resolve_model_grid_shape(HOPA_MODEL_PATH, hopa_network, test_files)
    if hops_shape is None or hopa_shape is None:
        print("Could not resolve model image shape from metadata or test folder. Retrain with option 3.")
        return
    if hops_shape != hopa_shape:
        print("Model shape mismatch between HOPS and HOPA. Retrain both models with option 3.")
        return

    model_shape = hops_shape
    print("Capturing HOPA intermediate stages pixel-by-pixel until stable state...")

    hops_recalled_grids: list[np.ndarray] = []
    hopa_recalled_grids: list[np.ndarray] = []
    hopa_intermediate_grids: list[np.ndarray] = []
    valid_files: list[Path] = []
    recall_rng = np.random.default_rng(42)

    for test_file in test_files:
        grid_array = load_binary_png_any_size(test_file)
        if grid_array is None:
            print(f"Skipping {test_file.name}: unable to read as a 2D pattern PNG.")
            continue
        if tuple(grid_array.shape) != model_shape:
            print(
                f"Skipping {test_file.name}: size {grid_array.shape[0]}x{grid_array.shape[1]} does not match "
                f"trained model size {model_shape[0]}x{model_shape[1]}."
            )
            continue

        vector = np.where(grid_array.reshape(-1) == 1, 1, -1)
        hops_recalled = hops_network.recall_synchronous(vector, steps=1)
        hopa_stages = recall_asynchronous_pixel_stages_until_stable(
            hopa_network,
            vector,
            rng=recall_rng,
        )
        hopa_recalled = hopa_stages[-1]
        hops_recalled_grids.append(bipolar_vector_to_grid(hops_recalled, model_shape))
        hopa_recalled_grids.append(bipolar_vector_to_grid(hopa_recalled, model_shape))
        hopa_stage_stack = np.stack(
            [bipolar_vector_to_grid(stage, model_shape) for stage in hopa_stages],
            axis=0,
        )
        hopa_intermediate_grids.append(hopa_stage_stack)
        valid_files.append(test_file)

    if not hops_recalled_grids:
        print("No valid patterns to recall.")
        return

    save_recent_recall_snapshot(test_folder, valid_files, hops_recalled_grids, hopa_recalled_grids, model_shape)
    print(f"Saved latest recall snapshot for Option 5: {LAST_RECALL_SNAPSHOT_PATH.name}")

    saved_hops = save_recalled_patterns(valid_files, hops_recalled_grids, "HOPS")
    saved_hopa = save_recalled_patterns(valid_files, hopa_recalled_grids, "HOPA")
    print(f"Saved recalled images: HOPS={saved_hops}, HOPA={saved_hopa} in {ensure_recall_patterns_dir()}")

    temp_run_folder = save_hopa_intermediate_stages(
        test_folder,
        valid_files,
        hopa_intermediate_grids,
        model_shape,
    )
    if temp_run_folder is not None:
        print(f"Saved HOPA intermediate stages in: {temp_run_folder}")
        print(f"Latest HOPA stage snapshot saved: {LAST_HOPA_STAGES_PATH.name}")

    display_recalled_patterns(valid_files, hops_recalled_grids, "HOPS")
    display_recalled_patterns(valid_files, hopa_recalled_grids, "HOPA")


def infer_reference_pattern_stem(test_file_stem: str) -> str:
    """Infer source pattern stem; noisy files use nXX_<source_stem> naming."""
    if test_file_stem.startswith("n") and "_" in test_file_stem:
        return test_file_stem.split("_", 1)[1]
    return test_file_stem


def format_error_cell(incorrect_count: int, incorrect_percent: float) -> str:
    """Format one cell as count and percent for terminal table output."""
    return f"{incorrect_count} ({incorrect_percent:.1f}%)"


def read_repeat_count() -> int:
    """Read how many repeated recall runs to execute for Option 6."""
    while True:
        value = input("Number of repeats [10]: ").strip()
        if value == "":
            return 10
        if value.isdigit() and int(value) > 0:
            return int(value)
        print("Invalid input: enter a positive integer.")


def read_monte_carlo_run_count() -> int:
    """Read how many Monte Carlo runs to execute for Option 7."""
    while True:
        value = input("Number of Monte Carlo runs [30]: ").strip()
        if value == "":
            return 30
        if value.isdigit() and int(value) > 0:
            return int(value)
        print("Invalid input: enter a positive integer.")


def _read_activation_index(prompt: str, allowed: set[int], default_value: int) -> int:
    """Read activation index from allowed set."""
    while True:
        value = input(prompt).strip().lower()
        if value == "":
            return default_value
        if value in {"1", "sign", "sin"} and 1 in allowed:
            return 1
        if value in {"2", "tanh", "tan"} and 2 in allowed:
            return 2
        if value in {"3", "softmax", "soft", "soft_max"} and 3 in allowed:
            return 3
        allowed_text = ", ".join(str(item) for item in sorted(allowed))
        print(f"Invalid activation choice. Allowed: {allowed_text}.")


def read_monte_carlo_activation_choices() -> list[str]:
    """Read activation subset selection for Option 7 (one, two, or all)."""
    activation_map = {1: "sign", 2: "tanh", 3: "softmax"}

    while True:
        print("Activation selection for Monte Carlo:")
        print("1. One activation")
        print("2. Two activations")
        print("3. All activations")
        selection = input("Choose [3]: ").strip().lower()

        if selection == "":
            selection = "3"

        if selection in {"3", "all"}:
            return ["sign", "tanh", "softmax"]

        if selection in {"1", "one", "single"}:
            print("Activation choices: 1=Sign, 2=Tanh, 3=SoftMax")
            chosen = _read_activation_index("Pick one [1]: ", {1, 2, 3}, 1)
            return [activation_map[chosen]]

        if selection in {"2", "two", "pair"}:
            print("Activation choices: 1=Sign, 2=Tanh, 3=SoftMax")
            first = _read_activation_index("Pick first [1]: ", {1, 2, 3}, 1)
            remaining = {1, 2, 3} - {first}
            second_default = min(remaining)
            second = _read_activation_index("Pick second: ", remaining, second_default)
            chosen_order = [first, second]
            return [activation_map[index] for index in chosen_order]

        print("Invalid choice. Enter 1 (one), 2 (two), or 3 (all).")


def _read_learning_mode_index(prompt: str, allowed: set[int], default_value: int) -> int:
    """Read learning-mode index from allowed set."""
    while True:
        value = input(prompt).strip().lower()
        if value == "":
            return default_value
        if value in {"1", "hebbian", "hebb"} and 1 in allowed:
            return 1
        if value in {"2", "storkey", "stork"} and 2 in allowed:
            return 2
        if value in {"3", "pseudo_inv", "pseudo-inverse", "pseudoinverse", "pinv"} and 3 in allowed:
            return 3
        allowed_text = ", ".join(str(item) for item in sorted(allowed))
        print(f"Invalid learning choice. Allowed: {allowed_text}.")


def read_monte_carlo_learning_choices() -> list[str]:
    """Read learning-mode subset selection for Option 7 (one, two, or all)."""
    learning_map = {1: "hebbian", 2: "storkey", 3: "pseudo_inv"}

    while True:
        print("Learning mode selection for Monte Carlo:")
        print("1. One learning mode")
        print("2. Two learning modes")
        print("3. All learning modes")
        selection = input("Choose [3]: ").strip().lower()

        if selection == "":
            selection = "3"

        if selection in {"3", "all"}:
            return ["hebbian", "storkey", "pseudo_inv"]

        if selection in {"1", "one", "single"}:
            print("Learning choices: 1=Hebbian, 2=Storkey, 3=Pseudo Inverse")
            chosen = _read_learning_mode_index("Pick one [1]: ", {1, 2, 3}, 1)
            return [learning_map[chosen]]

        if selection in {"2", "two", "pair"}:
            print("Learning choices: 1=Hebbian, 2=Storkey, 3=Pseudo Inverse")
            first = _read_learning_mode_index("Pick first [1]: ", {1, 2, 3}, 1)
            remaining = {1, 2, 3} - {first}
            second_default = min(remaining)
            second = _read_learning_mode_index("Pick second: ", remaining, second_default)
            chosen_order = [first, second]
            return [learning_map[index] for index in chosen_order]

        print("Invalid choice. Enter 1 (one), 2 (two), or 3 (all).")


def read_monte_carlo_compare_mode() -> str:
    """Read Option 7 compare mode submenu choice."""
    while True:
        print("Monte Carlo compare mode:")
        print("1. Compare Activation")
        print("2. Compare Learning")
        selection = input("Choose [1]: ").strip().lower()
        if selection == "":
            return "activation"
        if selection in {"1", "activation", "act"}:
            return "activation"
        if selection in {"2", "learning", "learn", "lm"}:
            return "learning"
        print("Invalid choice. Enter 1 (Compare Activation) or 2 (Compare Learning).")


def read_monte_carlo_recall_mode() -> str:
    """Read Option 7 recall method choice (HOPS or HOPA)."""
    while True:
        print("Recall method:")
        print("1. HOPS (synchronous)")
        print("2. HOPA (asynchronous)")
        selection = input("Choose [2]: ").strip().lower()
        if selection == "":
            return "HOPA"
        if selection in {"1", "hops", "sync", "synchronous"}:
            return "HOPS"
        if selection in {"2", "hopa", "async", "asynchronous"}:
            return "HOPA"
        print("Invalid choice. Enter 1 (HOPS) or 2 (HOPA).")


def _ci95_half_width(values: np.ndarray) -> float:
    """Return 95% CI half-width for sample mean using normal approximation."""
    n = int(values.size)
    if n <= 1:
        return 0.0
    sample_std = float(np.std(values, ddof=1))
    return 1.96 * sample_std / sqrt(float(n))


def _learning_mode_abbrev(learning_mode: str) -> str:
    """Return short learning-mode label for compact report headers."""
    mapping = {
        "hebbian": "HEB",
        "storkey": "STO",
        "pseudo_inv": "PI",
    }
    return mapping.get(learning_mode, learning_mode.upper())


def _safe_div(numerator: float, denominator: float) -> float:
    """Return numerator/denominator, or 0 when denominator is 0."""
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _binary_confusion_counts(reference: np.ndarray, predicted: np.ndarray) -> tuple[int, int, int, int]:
    """Return TP, FP, FN, TN counts for binary grids where 1 is the positive class."""
    tp = int(np.count_nonzero((predicted == 1) & (reference == 1)))
    fp = int(np.count_nonzero((predicted == 1) & (reference == 0)))
    fn = int(np.count_nonzero((predicted == 0) & (reference == 1)))
    tn = int(np.count_nonzero((predicted == 0) & (reference == 0)))
    return tp, fp, fn, tn


def _format_float_cell(value: float, width: int = 10) -> str:
    """Format metric values with fixed width for terminal table readability."""
    return f"{value:>{width}.4f}"


def run_repeat_recall_report() -> None:
    """Run repeated recalls and print aggregate error and average metric report in terminal."""
    print("\n=== Repeat Recall Report ===")
    default_test_folder = Path(__file__).resolve().parent / DEFAULT_RECALL_TEST_FOLDER
    test_folder = prompt_for_folder("hopfield.test_folder", "Folder to test", default_test_folder)

    if not test_folder.exists() or not test_folder.is_dir():
        print(f"Folder not found: {test_folder}")
        return

    repeat_count = read_repeat_count()

    hops_network = load_network_from_file(HOPS_MODEL_PATH)
    if hops_network is None:
        print("Trained HOPS model not found or invalid. Run option 3 first.")
        return

    hopa_network = load_network_from_file(HOPA_MODEL_PATH)
    if hopa_network is None:
        print("Trained HOPA model not found or invalid. Run option 3 first.")
        return

    test_files = sorted(
        test_folder.glob("*.png"),
        key=lambda file_path: file_path.stat().st_mtime,
        reverse=True,
    )[:8]

    if not test_files:
        print(f"No PNG images found in: {test_folder}")
        return

    hops_shape = resolve_model_grid_shape(HOPS_MODEL_PATH, hops_network, test_files)
    hopa_shape = resolve_model_grid_shape(HOPA_MODEL_PATH, hopa_network, test_files)
    if hops_shape is None or hopa_shape is None:
        print("Could not resolve model image shape from metadata or test folder. Retrain with option 3.")
        return
    if hops_shape != hopa_shape:
        print("Model shape mismatch between HOPS and HOPA. Retrain both models with option 3.")
        return

    model_shape = hops_shape

    patterns_folder = ensure_patterns_dir()
    test_cases: list[tuple[str, np.ndarray, np.ndarray]] = []

    for test_file in test_files:
        noisy_grid = load_binary_png_any_size(test_file)
        if noisy_grid is None:
            print(f"Skipping {test_file.name}: unable to read as a 2D pattern PNG.")
            continue
        if tuple(noisy_grid.shape) != model_shape:
            print(
                f"Skipping {test_file.name}: size {noisy_grid.shape[0]}x{noisy_grid.shape[1]} does not match "
                f"trained model size {model_shape[0]}x{model_shape[1]}."
            )
            continue

        reference_stem = infer_reference_pattern_stem(test_file.stem)
        reference_path = patterns_folder / f"{reference_stem}.png"
        reference_grid = load_binary_png_any_size(reference_path)
        if reference_grid is None:
            print(f"Skipping {test_file.name}: reference pattern not found/invalid ({reference_path.name}).")
            continue
        if tuple(reference_grid.shape) != model_shape:
            print(
                f"Skipping {test_file.name}: reference size {reference_grid.shape[0]}x{reference_grid.shape[1]} does not match "
                f"trained model size {model_shape[0]}x{model_shape[1]}."
            )
            continue

        test_cases.append(
            (
                test_file.name,
                np.where(noisy_grid.reshape(-1) == 1, 1, -1),
                reference_grid,
            )
        )

    if not test_cases:
        print("No valid files to report.")
        return

    model_totals: dict[str, dict[str, float]] = {
        "HOPA": {"errors": 0.0, "tp": 0.0, "fp": 0.0, "fn": 0.0, "tn": 0.0},
        "HOPS": {"errors": 0.0, "tp": 0.0, "fp": 0.0, "fn": 0.0, "tn": 0.0},
    }
    per_repeat_errors: dict[str, list[float]] = {"HOPA": [], "HOPS": []}

    for _ in range(repeat_count):
        hops_repeat_errors = 0.0
        hopa_repeat_errors = 0.0

        for _, noisy_vector, reference_array in test_cases:
            hops_recalled = hops_network.recall_synchronous(noisy_vector, steps=1)
            hopa_recalled = hopa_network.recall_asynchronous(noisy_vector, steps=1)

            hops_grid = bipolar_vector_to_grid(hops_recalled, model_shape)
            hopa_grid = bipolar_vector_to_grid(hopa_recalled, model_shape)

            hops_errors = int(np.count_nonzero(hops_grid != reference_array))
            hopa_errors = int(np.count_nonzero(hopa_grid != reference_array))

            model_totals["HOPS"]["errors"] += hops_errors
            model_totals["HOPA"]["errors"] += hopa_errors
            hops_repeat_errors += hops_errors
            hopa_repeat_errors += hopa_errors

            hops_tp, hops_fp, hops_fn, hops_tn = _binary_confusion_counts(reference_array, hops_grid)
            hopa_tp, hopa_fp, hopa_fn, hopa_tn = _binary_confusion_counts(reference_array, hopa_grid)

            model_totals["HOPS"]["tp"] += hops_tp
            model_totals["HOPS"]["fp"] += hops_fp
            model_totals["HOPS"]["fn"] += hops_fn
            model_totals["HOPS"]["tn"] += hops_tn

            model_totals["HOPA"]["tp"] += hopa_tp
            model_totals["HOPA"]["fp"] += hopa_fp
            model_totals["HOPA"]["fn"] += hopa_fn
            model_totals["HOPA"]["tn"] += hopa_tn

        per_repeat_errors["HOPS"].append(hops_repeat_errors)
        per_repeat_errors["HOPA"].append(hopa_repeat_errors)

    total_images_evaluated = repeat_count * len(test_cases)
    total_pixels_evaluated = total_images_evaluated * (model_shape[0] * model_shape[1])

    print(f"Repeats: {repeat_count}")
    print(f"Files used: {len(test_cases)} (up to 8)")
    print(f"Images evaluated: {total_images_evaluated}")
    print(f"Pixels evaluated per model: {total_pixels_evaluated}")
    print(f"Test Folder: {test_folder}")
    print()

    model_col_width = len("Model")
    avg_err_col_title = "Avg. ± SD"
    avg_err_col_width = len(avg_err_col_title)

    avg_error_texts: dict[str, str] = {}
    for model_name in ("HOPA", "HOPS"):
        repeat_errors = np.asarray(per_repeat_errors[model_name], dtype=float)
        avg_errors = float(np.mean(repeat_errors))
        std_errors = float(np.std(repeat_errors))
        avg_text = f"{avg_errors:.2f} ± {std_errors:.2f}"
        avg_error_texts[model_name] = avg_text
        avg_err_col_width = max(avg_err_col_width, len(avg_text))

    header = (
        f"{'Model':<{model_col_width}}  "
        f"{avg_err_col_title:>{avg_err_col_width}}  "
        f"{'P':>10}  {'R':>10}  {'S':>10}  {'F':>10}"
    )
    print(header)
    print("-" * len(header))

    for model_name in ("HOPA", "HOPS"):
        totals = model_totals[model_name]
        tp = totals["tp"]
        fp = totals["fp"]
        fn = totals["fn"]
        tn = totals["tn"]

        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        specificity = _safe_div(tn, tn + fp)
        f_score = _safe_div(2 * precision * recall, precision + recall)
        avg_error_text = avg_error_texts[model_name]

        print(
            f"{model_name:<{model_col_width}}  "
            f"{avg_error_text:>{avg_err_col_width}}  "
            f"{_format_float_cell(precision)}  "
            f"{_format_float_cell(recall)}  "
            f"{_format_float_cell(specificity)}  "
            f"{_format_float_cell(f_score)}"
        )


def run_monte_carlo_report() -> None:
    """Run Monte Carlo recall and compare activation or learning mode with mean +/- 95% CI errors."""
    print("\n=== Monte Carlo Recall Report ===")
    print("Seed policy: base seed = 42")

    training_folder = prompt_for_folder(
        "hopfield.training_folder",
        "Training folder",
        ensure_patterns_dir(),
    )
    if not training_folder.exists() or not training_folder.is_dir():
        print(f"Folder not found: {training_folder}")
        return

    test_folder = prompt_for_folder(
        "hopfield.test_folder",
        "Folder to test",
        Path(__file__).resolve().parent / DEFAULT_RECALL_TEST_FOLDER,
    )
    if not test_folder.exists() or not test_folder.is_dir():
        print(f"Folder not found: {test_folder}")
        return

    reference_folder = training_folder

    compare_mode = read_monte_carlo_compare_mode()

    fixed_learning_mode: str | None = None
    fixed_activation: str | None = None
    comparison_labels: list[str]

    if compare_mode == "activation":
        fixed_learning_mode = read_learning_mode_choice()
        comparison_labels = read_monte_carlo_activation_choices()
    else:
        fixed_activation = read_activation_choice()
        comparison_labels = read_monte_carlo_learning_choices()

    recall_mode = read_monte_carlo_recall_mode()

    run_count = read_monte_carlo_run_count()

    vectors, used_files, grid_shape = load_training_patterns(training_folder)
    if not vectors:
        print("No valid training patterns found. Monte Carlo aborted.")
        return
    if grid_shape is None:
        print("Could not determine training dimensions. Monte Carlo aborted.")
        return

    if compare_mode == "activation":
        print(
            f"Preparing models: compare=activation, learning={fixed_learning_mode}, "
            f"activations={', '.join(comparison_labels)}, patterns={len(used_files)}, "
            f"grid={grid_shape[0]}x{grid_shape[1]}"
        )
    else:
        print(
            f"Preparing models: compare=learning, activation={fixed_activation}, "
            f"learning_modes={', '.join(comparison_labels)}, patterns={len(used_files)}, "
            f"grid={grid_shape[0]}x{grid_shape[1]}"
        )

    neuron_count = int(vectors[0].size)
    networks: dict[str, HopfieldNetwork] = {}
    for label in comparison_labels:
        if compare_mode == "activation":
            activation_name = label
            learning_mode = fixed_learning_mode if fixed_learning_mode is not None else DEFAULT_LEARNING_MODE
            print(f"Training activation '{activation_name}'...")
            network = HopfieldNetwork(neuron_count, activation=activation_name)
            train_network(network, vectors, learning_mode, label=f"MC-{activation_name.upper()}")
        else:
            learning_mode = label
            activation_name = fixed_activation if fixed_activation is not None else DEFAULT_ACTIVATION
            print(f"Training learning mode '{learning_mode}'...")
            network = HopfieldNetwork(neuron_count, activation=activation_name)
            train_network(network, vectors, learning_mode, label=f"MC-{_learning_mode_abbrev(learning_mode)}")
        networks[label] = network

    test_files = sorted(
        test_folder.glob("*.png"),
        key=lambda file_path: file_path.stat().st_mtime,
        reverse=True,
    )[:8]

    if not test_files:
        print(f"No PNG images found in: {test_folder}")
        return

    test_cases: list[tuple[str, np.ndarray, np.ndarray]] = []

    for test_file in test_files:
        noisy_grid = load_binary_png_any_size(test_file)
        if noisy_grid is None:
            print(f"Skipping {test_file.name}: unable to read as a 2D pattern PNG.")
            continue
        if tuple(noisy_grid.shape) != grid_shape:
            print(
                f"Skipping {test_file.name}: size {noisy_grid.shape[0]}x{noisy_grid.shape[1]} does not match "
                f"model size {grid_shape[0]}x{grid_shape[1]}."
            )
            continue

        reference_stem = infer_reference_pattern_stem(test_file.stem)
        reference_path = reference_folder / f"{reference_stem}.png"
        reference_grid = load_binary_png_any_size(reference_path)
        if reference_grid is None:
            print(f"Skipping {test_file.name}: reference pattern not found/invalid ({reference_path.name}).")
            continue
        if tuple(reference_grid.shape) != grid_shape:
            print(
                f"Skipping {test_file.name}: reference size {reference_grid.shape[0]}x{reference_grid.shape[1]} does not match "
                f"model size {grid_shape[0]}x{grid_shape[1]}."
            )
            continue

        test_cases.append(
            (
                test_file.name,
                np.where(noisy_grid.reshape(-1) == 1, 1, -1),
                reference_grid,
            )
        )

    if not test_cases:
        print("No valid files to evaluate.")
        return

    print(
        f"Running Monte Carlo: runs={run_count}, files={len(test_cases)}, "
        f"columns={len(comparison_labels)}"
    )

    row_names = [file_name for file_name, _, _ in test_cases]
    row_names.append("FOLDER_TOTAL")
    errors_by_column: dict[str, dict[str, list[float]]] = {
        label: {row_name: [] for row_name in row_names}
        for label in comparison_labels
    }

    for run_index in range(run_count):
        print(f"[MC] Run {run_index + 1}/{run_count}")
        for label_index, label in enumerate(comparison_labels):
            run_seed = 42 + (run_index * 1000) + label_index
            run_rng = np.random.default_rng(run_seed)
            run_total_errors = 0

            for file_name, noisy_vector, reference_grid in test_cases:
                if recall_mode == "HOPS":
                    recalled = networks[label].recall_synchronous(noisy_vector, steps=1)
                else:
                    recalled = networks[label].recall_asynchronous(noisy_vector, steps=1, rng=run_rng)
                recalled_grid = bipolar_vector_to_grid(recalled, grid_shape)
                errors = int(np.count_nonzero(recalled_grid != reference_grid))
                errors_by_column[label][file_name].append(float(errors))
                run_total_errors += errors

            errors_by_column[label]["FOLDER_TOTAL"].append(float(run_total_errors))
            label_text = label.upper() if compare_mode == "activation" else _learning_mode_abbrev(label)
            print(
                f"  [MC] {label_text:<7} complete "
                f"(seed={run_seed}, total_errors={run_total_errors})"
            )

    ci_title = "Mean +/- 95% CI"
    row_width = max(len("Pattern"), max(len(row_name) for row_name in row_names))
    column_widths: dict[str, int] = {}

    formatted_cells: dict[str, dict[str, str]] = {row_name: {} for row_name in row_names}
    for label in comparison_labels:
        column_title = label.upper() if compare_mode == "activation" else _learning_mode_abbrev(label)
        column_widths[label] = max(len(column_title), len(ci_title))
        for row_name in row_names:
            row_values = np.asarray(errors_by_column[label][row_name], dtype=float)
            mean_value = float(np.mean(row_values))
            ci_half_width = _ci95_half_width(row_values)
            cell_text = f"{mean_value:.2f} +/- {ci_half_width:.2f}"
            formatted_cells[row_name][label] = cell_text
            column_widths[label] = max(column_widths[label], len(cell_text))

    header_parts = [f"{'Pattern':<{row_width}}"]
    for label in comparison_labels:
        column_title = label.upper() if compare_mode == "activation" else _learning_mode_abbrev(label)
        header_parts.append(f"{column_title:>{column_widths[label]}}")
    header = "  ".join(header_parts)

    print()
    print()
    if compare_mode == "activation":
        header_variant = _learning_mode_abbrev(
            fixed_learning_mode if fixed_learning_mode is not None else DEFAULT_LEARNING_MODE
        )
    else:
        header_variant = (fixed_activation if fixed_activation is not None else DEFAULT_ACTIVATION).upper()

    title_main = f"MONTE CARLO ERROR TABLE - {recall_mode} - {header_variant} - {run_count}"
    if compare_mode == "activation":
        fixed_text = f"LM={_learning_mode_abbrev(fixed_learning_mode if fixed_learning_mode is not None else DEFAULT_LEARNING_MODE)}"
        compare_text = "CMP=ACT"
    else:
        fixed_text = f"ACT={(fixed_activation if fixed_activation is not None else DEFAULT_ACTIVATION).upper()}"
        compare_text = "CMP=LRN"
    title_info = f"{compare_text} | RCL={recall_mode} | {fixed_text} | RUNS={run_count} | FILES={len(test_cases)}"
    title_border = "=" * max(len(title_main), len(title_info), len(header))

    print(title_border)
    print(title_main)
    print(title_info)
    print(title_border)
    print(f"Columns: {ci_title}")
    print(header)
    print("-" * len(header))

    for row_name in row_names:
        row_parts = [f"{row_name:<{row_width}}"]
        for label in comparison_labels:
            row_parts.append(f"{formatted_cells[row_name][label]:>{column_widths[label]}}")
        print("  ".join(row_parts))

    print("-" * len(header))
    print("Monte Carlo run complete.")

def run_recall_error_report() -> None:
    """Print per-file recall error counts/percentages from the latest snapshot."""
    print("\n=== Recall Error Report ===")
    snapshot = load_recent_recall_snapshot()
    if snapshot is None:
        print("No recall snapshot available. Run option 4 first.")
        return

    hops_meta = load_model_metadata(HOPS_MODEL_PATH)
    hopa_meta = load_model_metadata(HOPA_MODEL_PATH)

    test_folder = Path(str(snapshot["test_folder"]))
    file_names = cast(list[str], snapshot["file_names"])
    hops_recalled = np.asarray(snapshot["hops_recalled"], dtype=np.uint8)
    hopa_recalled = np.asarray(snapshot["hopa_recalled"], dtype=np.uint8)
    snapshot_rows = int(cast(int | str, snapshot["grid_rows"]))
    snapshot_cols = int(cast(int | str, snapshot["grid_cols"]))
    snapshot_shape = (snapshot_rows, snapshot_cols)

    default_reference_folder = ensure_patterns_dir()
    patterns_folder = prompt_for_folder(
        "hopfield.reference_folder",
        "Reference pattern folder",
        default_reference_folder,
    )

    if not patterns_folder.exists() or not patterns_folder.is_dir():
        print(f"Folder not found: {patterns_folder}")
        return

    rows: list[tuple[str, str, str]] = []
    total_hopa_incorrect = 0
    total_hops_incorrect = 0

    for index, file_name in enumerate(file_names):
        reference_stem = infer_reference_pattern_stem(Path(file_name).stem)
        reference_path = patterns_folder / f"{reference_stem}.png"
        reference_grid = load_binary_png_any_size(reference_path)
        if reference_grid is None:
            print(f"Skipping {file_name}: reference pattern not found/invalid ({reference_path.name}).")
            continue
        if tuple(reference_grid.shape) != snapshot_shape:
            print(
                f"Skipping {file_name}: reference size {reference_grid.shape[0]}x{reference_grid.shape[1]} does not match "
                f"snapshot size {snapshot_shape[0]}x{snapshot_shape[1]}."
            )
            continue

        hops_recalled_grid = np.asarray(hops_recalled[index], dtype=np.uint8)
        hopa_recalled_grid = np.asarray(hopa_recalled[index], dtype=np.uint8)

        reference_array = reference_grid
        hops_incorrect = int(np.count_nonzero(hops_recalled_grid != reference_array))
        hopa_incorrect = int(np.count_nonzero(hopa_recalled_grid != reference_array))

        neuron_count = int(reference_array.size)
        hops_percent = (hops_incorrect / neuron_count) * 100.0
        hopa_percent = (hopa_incorrect / neuron_count) * 100.0
        total_hopa_incorrect += hopa_incorrect
        total_hops_incorrect += hops_incorrect

        rows.append(
            (
                file_name,
                format_error_cell(hopa_incorrect, hopa_percent),
                format_error_cell(hops_incorrect, hops_percent),
            )
        )

    if not rows:
        print("No valid files to report.")
        return

    rows.sort(key=lambda row: row[0].lower())

    total_hopa_text = str(total_hopa_incorrect)
    total_hops_text = str(total_hops_incorrect)

    file_col_width = max(len("File"), len("TOTAL"), max(len(row[0]) for row in rows))
    hopa_col_width = max(len("HOPA"), len(total_hopa_text), max(len(row[1]) for row in rows))
    hops_col_width = max(len("HOPS"), len(total_hops_text), max(len(row[2]) for row in rows))

    print("NN Status")
    if hops_meta is not None:
        print(
            "HOPS: trained "
            f"neurons={hops_meta['size']} "
            f"activation={hops_meta['activation']} "
            f"learning={hops_meta['learning_mode']} "
            f"patterns={hops_meta['pattern_count']}"
        )
    else:
        print("HOPS: trained (metadata unavailable)")

    if hopa_meta is not None:
        print(
            "HOPA: trained "
            f"neurons={hopa_meta['size']} "
            f"activation={hopa_meta['activation']} "
            f"learning={hopa_meta['learning_mode']} "
            f"patterns={hopa_meta['pattern_count']}"
        )
    else:
        print("HOPA: trained (metadata unavailable)")

    print(f"Test Folder: {test_folder}")
    print(f"Reference Folder: {patterns_folder}")

    header = f"{'File':<{file_col_width}}  {'HOPA':>{hopa_col_width}}  {'HOPS':>{hops_col_width}}"
    print(header)
    print("-" * len(header))
    for file_name, hopa_text, hops_text in rows:
        print(f"{file_name:<{file_col_width}}  {hopa_text:>{hopa_col_width}}  {hops_text:>{hops_col_width}}")
    print("-" * len(header))
    print(f"{'TOTAL':<{file_col_width}}  {total_hopa_text:<{hopa_col_width}}  {total_hops_text:<{hops_col_width}}")


def run_hopfield_training() -> None:
    """Train and save synchronous/asynchronous Hopfield models from PNG patterns."""
    print("\n=== Hopfield Training ===")
    default_folder = ensure_patterns_dir()
    training_folder = prompt_for_folder("hopfield.training_folder", "Training folder", default_folder)

    if not training_folder.exists() or not training_folder.is_dir():
        print(f"Folder not found: {training_folder}")
        return

    vectors, used_files, grid_shape = load_training_patterns(training_folder)
    if not vectors:
        print("No valid training patterns found. Training aborted.")
        return

    if grid_shape is None:
        print("Could not determine training dimensions. Training aborted.")
        return

    vector_size = int(vectors[0].size)
    neuron_count = vector_size
    activation_name = read_activation_choice()
    learning_mode = read_learning_mode_choice()

    print(f"Starting training with {len(vectors)} pattern(s), vector size {vector_size}...")
    print(f"Detected grid size: {grid_shape[0]}x{grid_shape[1]} -> neurons={neuron_count}")
    print(f"Configuration: activation={activation_name}, learning={learning_mode}")

    sync_network = HopfieldNetwork(neuron_count, activation=activation_name)
    train_network(sync_network, vectors, learning_mode, label="HOPS")
    save_network_params(
        HOPS_MODEL_PATH,
        "HOPS",
        sync_network,
        training_folder,
        len(used_files),
        learning_mode,
        grid_shape,
    )

    async_network = HopfieldNetwork(neuron_count, activation=activation_name)
    train_network(async_network, vectors, learning_mode, label="HOPA")
    save_network_params(
        HOPA_MODEL_PATH,
        "HOPA",
        async_network,
        training_folder,
        len(used_files),
        learning_mode,
        grid_shape,
    )

    sample = vectors[0]
    sync_recalled = sync_network.recall_synchronous(sample, steps=1)
    async_recalled = async_network.recall_asynchronous(sample, steps=1)
    sync_match = int(np.array_equal(sync_recalled, sample))
    async_match = int(np.array_equal(async_recalled, sample))

    print(f"[SYNC] Sample recall exact match: {'yes' if sync_match else 'no'}")
    print(f"[ASYNC] Sample recall exact match: {'yes' if async_match else 'no'}")

    sync_energies = track_energy_synchronous(sync_network, sample, steps=5)
    async_energies = track_energy_asynchronous(async_network, sample, steps=5)
    sync_energy_text = " -> ".join(f"{energy_value:.2f}" for energy_value in sync_energies)
    async_energy_text = " -> ".join(f"{energy_value:.2f}" for energy_value in async_energies)
    print(f"[SYNC] Energy track: {sync_energy_text}")
    print(f"[ASYNC] Energy track: {async_energy_text}")

    print(
        "Hopfield training completed successfully "
        f"for {len(used_files)} pattern(s) from '{training_folder.name}'."
    )


if __name__ == "__main__":
    install_terminal_output_logger()
    run_hopfield_training()
