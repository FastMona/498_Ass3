from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from multiprocessing import get_context

from create_img import GRID_ROWS, GRID_COLS, ensure_patterns_dir, load_pattern_image
from terminal_out import install_terminal_output_logger


MODELS_DIR = Path(__file__).resolve().parent / "nn_models"
HOPS_MODEL_PATH = MODELS_DIR / "HOPS.npz"
HOPA_MODEL_PATH = MODELS_DIR / "HOPA.npz"
LAST_RECALL_SNAPSHOT_PATH = MODELS_DIR / "LAST_RECALL_SNAPSHOT.npz"
NEURON_COUNT = GRID_ROWS * GRID_COLS
DEFAULT_ACTIVATION = "tan"
DEFAULT_LEARNING_MODE = "hebbian"
DEFAULT_RECALL_TEST_FOLDER = "noisy_patterns"


def _set_window_title(fig, window_title: str | None) -> None:
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
        if self.activation == "tan":
            activated = np.tanh(values)
            return np.where(activated >= 0, 1, -1)
        if self.activation == "sin":
            activated = np.sin(values)
            return np.where(activated >= 0, 1, -1)
        return np.where(values >= 0, 1, -1)

    def train_hebbian(self, patterns: list[np.ndarray], label: str = "HEBB") -> None:
        """Batch Hebbian update."""
        self.weights.fill(0)
        for index, pattern in enumerate(patterns, start=1):
            self.weights += np.outer(pattern, pattern)
            print(f"[{label}] Hebbian processed pattern {index}/{len(patterns)}")

        self.weights /= self.size
        np.fill_diagonal(self.weights, 0)
        print(f"[{label}] Hebbian training complete.")

    def train_storkey(self, patterns: list[np.ndarray], label: str = "STORK") -> None:
        """Storkey learning rule."""
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
        """Pseudo-inverse learning rule."""
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

    def recall_asynchronous(self, pattern: np.ndarray, steps: int = 1) -> np.ndarray:
        """Recall using asynchronous state updates."""
        state = pattern.copy()
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


def load_training_patterns(folder: Path) -> tuple[list[np.ndarray], list[Path]]:
    """Load valid training patterns from PNG files in a folder."""
    image_files = sorted(folder.glob("*.png"))
    vectors: list[np.ndarray] = []
    used_files: list[Path] = []

    if not image_files:
        return vectors, used_files

    print(f"Found {len(image_files)} PNG file(s). Validating {GRID_COLS}x{GRID_ROWS} patterns...")
    for index, image_file in enumerate(image_files, start=1):
        grid = load_pattern_image(image_file)
        if grid is None:
            print(f"Skipping {image_file.name}: invalid shape/format.")
            continue

        vectors.append(grid_to_bipolar_vector(grid))
        used_files.append(image_file)
        print(f"Loaded pattern {len(vectors)} from {image_file.name} ({index}/{len(image_files)})")

    return vectors, used_files


def read_activation_choice() -> str:
    """Prompt user for activation choice."""
    while True:
        value = input(f"Activation [tan/sin] (default {DEFAULT_ACTIVATION}): ").strip().lower()
        if value == "":
            return DEFAULT_ACTIVATION
        if value in {"tan", "sin"}:
            return value
        print("Invalid activation. Choose 'tan' or 'sin'.")


def read_learning_mode_choice() -> str:
    """Prompt user for learning mode."""
    while True:
        value = input(f"Learning mode [hebbian/storkey/pseudo_inv] (default {DEFAULT_LEARNING_MODE}): ").strip().lower()
        if value == "":
            return DEFAULT_LEARNING_MODE
        if value in {"hebbian", "storkey", "pseudo_inv"}:
            return value
        print("Invalid learning mode. Choose 'hebbian', 'storkey', or 'pseudo_inv'.")


def ensure_models_dir() -> Path:
    """Ensure model output folder exists and return its path."""
    MODELS_DIR.mkdir(exist_ok=True)
    return MODELS_DIR


def save_network_params(
    model_path: Path,
    model_name: str,
    network: HopfieldNetwork,
    training_folder: Path,
    pattern_count: int,
    learning_mode: str,
) -> None:
    """Persist network parameters so retraining is not required on next startup."""
    ensure_models_dir()
    np.savez(
        model_path,
        weights=network.weights,
        activation=network.activation,
        size=network.size,
        grid_rows=GRID_ROWS,
        grid_cols=GRID_COLS,
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


def bipolar_vector_to_grid(vector: np.ndarray) -> np.ndarray:
    """Convert bipolar vector (-1/+1) back to binary image grid (0/1)."""
    binary = np.where(vector.reshape(-1) == 1, 1, 0)
    return binary.reshape((GRID_ROWS, GRID_COLS))


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

    if size != NEURON_COUNT:
        return None
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

    learning_mode = str(data["learning_mode"]) if "learning_mode" in data else "unknown"
    pattern_count = str(int(data["pattern_count"])) if "pattern_count" in data else "unknown"
    training_folder = str(data["training_folder"]) if "training_folder" in data else "unknown"

    return {
        "size": str(size),
        "activation": activation,
        "learning_mode": learning_mode,
        "pattern_count": pattern_count,
        "training_folder": training_folder,
    }


def save_recent_recall_snapshot(
    test_folder: Path,
    valid_files: list[Path],
    hops_recalled_grids: list[np.ndarray],
    hopa_recalled_grids: list[np.ndarray],
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

    return {
        "test_folder": str(data["test_folder"]),
        "file_names": file_names,
        "hops_recalled": hops_recalled,
        "hopa_recalled": hopa_recalled,
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


def run_pattern_recall() -> None:
    """Recall up to 8 patterns from a test folder using both trained models."""
    print("\n=== Pattern Recall ===")
    folder_input = input(f"Folder to test [{DEFAULT_RECALL_TEST_FOLDER}]: ").strip()
    test_folder = Path(folder_input) if folder_input else Path(DEFAULT_RECALL_TEST_FOLDER)

    if not test_folder.is_absolute():
        test_folder = Path(__file__).resolve().parent / test_folder

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

    hops_recalled_grids: list[np.ndarray] = []
    hopa_recalled_grids: list[np.ndarray] = []
    valid_files: list[Path] = []

    for test_file in test_files:
        grid = load_pattern_image(test_file)
        if grid is None:
            print(f"Skipping {test_file.name}: expected {GRID_COLS}x{GRID_ROWS} binary pattern PNG.")
            continue

        vector = grid_to_bipolar_vector(grid)
        hops_recalled = hops_network.recall_synchronous(vector, steps=1)
        hopa_recalled = hopa_network.recall_asynchronous(vector, steps=1)
        hops_recalled_grids.append(bipolar_vector_to_grid(hops_recalled))
        hopa_recalled_grids.append(bipolar_vector_to_grid(hopa_recalled))
        valid_files.append(test_file)

    if not hops_recalled_grids:
        print("No valid patterns to recall.")
        return

    save_recent_recall_snapshot(test_folder, valid_files, hops_recalled_grids, hopa_recalled_grids)
    print(f"Saved latest recall snapshot for Option 5: {LAST_RECALL_SNAPSHOT_PATH.name}")

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


def run_recall_error_report() -> None:
    """Print per-file recalled pixel error count and percent using latest persisted recall snapshot."""
    print("\n=== Recall Error Report ===")
    snapshot = load_recent_recall_snapshot()
    if snapshot is None:
        print("No recall snapshot available. Run option 4 first.")
        return

    hops_meta = load_model_metadata(HOPS_MODEL_PATH)
    hopa_meta = load_model_metadata(HOPA_MODEL_PATH)

    test_folder = Path(str(snapshot["test_folder"]))
    file_names = list(snapshot["file_names"])
    hops_recalled = np.asarray(snapshot["hops_recalled"], dtype=np.uint8)
    hopa_recalled = np.asarray(snapshot["hopa_recalled"], dtype=np.uint8)

    patterns_folder = ensure_patterns_dir()
    rows: list[tuple[str, str, str]] = []

    for index, file_name in enumerate(file_names):
        reference_stem = infer_reference_pattern_stem(Path(file_name).stem)
        reference_path = patterns_folder / f"{reference_stem}.png"
        reference_grid = load_pattern_image(reference_path)
        if reference_grid is None:
            print(f"Skipping {file_name}: reference pattern not found/invalid ({reference_path.name}).")
            continue

        hops_recalled_grid = np.asarray(hops_recalled[index], dtype=np.uint8)
        hopa_recalled_grid = np.asarray(hopa_recalled[index], dtype=np.uint8)

        reference_array = np.asarray(reference_grid, dtype=np.uint8)
        hops_incorrect = int(np.count_nonzero(hops_recalled_grid != reference_array))
        hopa_incorrect = int(np.count_nonzero(hopa_recalled_grid != reference_array))

        hops_percent = (hops_incorrect / NEURON_COUNT) * 100.0
        hopa_percent = (hopa_incorrect / NEURON_COUNT) * 100.0

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

    file_col_width = max(len("File"), max(len(row[0]) for row in rows))
    hopa_col_width = max(len("HOPA"), max(len(row[1]) for row in rows))
    hops_col_width = max(len("HOPS"), max(len(row[2]) for row in rows))

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
    header = f"{'File':<{file_col_width}}  {'HOPA':>{hopa_col_width}}  {'HOPS':>{hops_col_width}}"
    print(header)
    print("-" * len(header))
    for file_name, hopa_text, hops_text in rows:
        print(f"{file_name:<{file_col_width}}  {hopa_text:>{hopa_col_width}}  {hops_text:>{hops_col_width}}")


def run_hopfield_training() -> None:
    """Prompt for folder and train Hopfield model in sync + async modes."""
    print("\n=== Hopfield Training ===")
    default_folder = ensure_patterns_dir()
    folder_input = input(f"Training folder [{default_folder.name}]: ").strip()
    training_folder = Path(folder_input) if folder_input else default_folder

    if not training_folder.is_absolute():
        training_folder = Path(__file__).resolve().parent / training_folder

    if not training_folder.exists() or not training_folder.is_dir():
        print(f"Folder not found: {training_folder}")
        return

    vectors, used_files = load_training_patterns(training_folder)
    if not vectors:
        print("No valid training patterns found. Training aborted.")
        return

    vector_size = GRID_ROWS * GRID_COLS
    if vector_size != NEURON_COUNT:
        print(
            "Configuration mismatch: "
            f"vector size is {vector_size} but NEURON_COUNT is {NEURON_COUNT}."
        )
        return

    neuron_count = NEURON_COUNT
    activation_name = read_activation_choice()
    learning_mode = read_learning_mode_choice()

    print(f"Starting training with {len(vectors)} pattern(s), vector size {vector_size}...")
    print(f"Configuration: activation={activation_name}, learning={learning_mode}")

    sync_network = HopfieldNetwork(neuron_count, activation=activation_name)
    train_network(sync_network, vectors, learning_mode, label="HOPS")
    save_network_params(HOPS_MODEL_PATH, "HOPS", sync_network, training_folder, len(used_files), learning_mode)

    async_network = HopfieldNetwork(neuron_count, activation=activation_name)
    train_network(async_network, vectors, learning_mode, label="HOPA")
    save_network_params(HOPA_MODEL_PATH, "HOPA", async_network, training_folder, len(used_files), learning_mode)

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
