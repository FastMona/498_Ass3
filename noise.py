from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from multiprocessing import get_context
import numpy as np

from create_img import ensure_patterns_dir
from folder_prefs import prompt_for_folder


NOISY_PATTERNS_DIR = Path(__file__).resolve().parent / "noisy_patterns"


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
        ),
        daemon=True,
    )
    process.start()


def ensure_noisy_patterns_dir() -> Path:
    """Ensure noisy patterns output folder exists and return its path."""
    NOISY_PATTERNS_DIR.mkdir(exist_ok=True)
    return NOISY_PATTERNS_DIR


def confirm_yes_no_default_no(prompt: str) -> bool:
    """Return True only when user explicitly answers yes; default is no."""
    response = input(f"{prompt} [y/N]: ").strip().lower()
    return response in {"y", "yes"}


def delete_png_files_in_folder(folder: Path) -> int:
    """Delete all PNG files in the given folder and return count deleted."""
    if not folder.exists() or not folder.is_dir():
        return 0

    deleted_count = 0
    for png_file in folder.glob("*.png"):
        png_file.unlink(missing_ok=True)
        deleted_count += 1
    return deleted_count


def read_noise_percent() -> float:
    """Read noise percent in range [0, 100]."""
    while True:
        value = input("Enter percent noise (0-100): ").strip()
        try:
            noise_percent = float(value)
        except ValueError:
            print("Invalid input: enter a number between 0 and 100.")
            continue

        if 0 <= noise_percent <= 100:
            return noise_percent
        print("Invalid input: percent noise must be between 0 and 100.")


def load_binary_png_any_size(image_path: Path) -> np.ndarray | None:
    """Load a PNG as a 2D binary grid (0/1) of its native size."""
    try:
        data = plt.imread(image_path)
    except OSError:
        return None

    if data.ndim == 3:
        data = data[..., :3].mean(axis=2)

    if data.ndim != 2:
        return None

    return (np.asarray(data) < 0.5).astype(np.uint8)


def apply_noise_to_grid(grid_array: np.ndarray, noise_percent: float) -> np.ndarray:
    """Randomly flip approximately noise_percent of bits in a binary grid."""
    if grid_array.ndim != 2:
        raise ValueError("Expected a 2D grid array.")

    total_bits = grid_array.size
    flip_count = int(round((noise_percent / 100.0) * total_bits))

    if flip_count <= 0:
        return grid_array.copy()

    rng = np.random.default_rng()
    flat_indices = rng.choice(total_bits, size=flip_count, replace=False)
    flat_grid = grid_array.reshape(-1).copy()
    flat_grid[flat_indices] = 1 - flat_grid[flat_indices]
    return flat_grid.reshape(grid_array.shape)


def display_recent_noisy_patterns(folder: Path, noise_percent_text: str) -> None:
    """Display a 2x4 figure of the 8 most recently generated noisy images."""
    recent_files = sorted(
        folder.glob("*.png"),
        key=lambda file_path: file_path.stat().st_mtime,
        reverse=True,
    )[:8]

    if not recent_files:
        return

    images: list[np.ndarray] = []
    image_titles: list[str] = []

    for image_path in recent_files:
        images.append(np.asarray(plt.imread(image_path)))
        image_titles.append(image_path.stem)

    ordered_items = sorted(zip(image_titles, images), key=lambda item: item[0].lower())
    image_titles = [item[0] for item in ordered_items]
    images = [item[1] for item in ordered_items]

    dimensions = {(image.shape[0], image.shape[1]) for image in images}
    if len(dimensions) == 1:
        rows, cols = next(iter(dimensions))
        size_text = f" ({rows} X {cols})"
    else:
        size_text = " (mixed sizes)"

    show_gallery_window(
        images,
        image_titles,
        suptitle=f"{noise_percent_text}% Noisy Patterns{size_text}",
        window_title="Most Recent Noisy Patterns",
    )


def run_create_noisy_patterns() -> None:
    """Create noisy versions of all patterns from a selected source folder."""
    default_folder = ensure_patterns_dir()
    source_folder = prompt_for_folder("noise.source_folder", "Folder to noisify", default_folder)

    if not source_folder.exists() or not source_folder.is_dir():
        print(f"Folder not found: {source_folder}")
        return

    noise_percent = read_noise_percent()
    if noise_percent.is_integer():
        noise_percent_text = str(int(noise_percent))
    else:
        noise_percent_text = str(noise_percent).rstrip("0").rstrip(".")

    if noise_percent.is_integer():
        noise_label = str(int(noise_percent))
    else:
        noise_label = str(noise_percent).rstrip("0").rstrip(".").replace(".", "p")
    destination_folder = ensure_noisy_patterns_dir()

    existing_noisy_images = list(destination_folder.glob("*.png"))
    if existing_noisy_images:
        if confirm_yes_no_default_no("Empty noisy_patterns folder before creating noisy images?"):
            deleted_count = delete_png_files_in_folder(destination_folder)
            print(f"Deleted {deleted_count} existing noisy image file(s) from {destination_folder.name}.")
        else:
            print(f"Keeping existing files in {destination_folder.name}.")

    source_images = sorted(source_folder.glob("*.png"))
    if not source_images:
        print(f"No PNG patterns found in: {source_folder}")
        return

    created_count = 0
    for source_image in source_images:
        grid_array = load_binary_png_any_size(source_image)
        if grid_array is None:
            print(f"Skipping {source_image.name}: unable to read as a 2D pattern PNG.")
            continue

        noisy_grid = apply_noise_to_grid(grid_array, noise_percent)
        output_path = destination_folder / f"n{noise_label}_{source_image.stem}.png"
        cmap = ListedColormap(["white", "black"])
        plt.imsave(output_path, noisy_grid, cmap=cmap, vmin=0, vmax=1)
        created_count += 1
        print(f"Created: {output_path.name}")

    print(f"Noisy pattern generation complete: {created_count} file(s) in {destination_folder}")

    if created_count <= 0:
        return

    display_recent_noisy_patterns(destination_folder, noise_percent_text)
