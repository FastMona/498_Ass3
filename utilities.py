from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from multiprocessing import get_context
import numpy as np

from create_img import ensure_patterns_dir
from create_img_folder import run_create_img_folder_utility
from folder_prefs import prompt_for_folder


def _workspace_dir() -> Path:
    return Path(__file__).resolve().parent


def _resolve_source_folder(default_folder: Path) -> Path:
    return prompt_for_folder("utilities.process_folder", "Folder to process", default_folder)


def _resolve_folder_input(default_folder: Path, prompt_label: str) -> Path:
    if prompt_label == "Folder to view":
        return prompt_for_folder("utilities.view_folder", prompt_label, default_folder)
    return prompt_for_folder("utilities.generic_folder", prompt_label, default_folder)


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


def _read_positive_int(prompt: str) -> int:
    while True:
        raw = input(prompt).strip()
        if raw.isdigit() and int(raw) > 0:
            return int(raw)
        print("Invalid input: enter a positive whole number.")


def _load_binary_png_any_size(image_path: Path) -> np.ndarray | None:
    try:
        data = plt.imread(image_path)
    except OSError:
        return None

    if data.ndim == 3:
        data = data[..., :3].mean(axis=2)

    if data.ndim != 2:
        return None

    return (np.asarray(data) < 0.5).astype(np.uint8)


def _format_dimension(value: int) -> str:
    if value < 100:
        return f"{value:02d}"
    return str(value)


def run_upsize_patterns_utility() -> None:
    """Create larger binary pattern images by padding right/bottom with blanks."""
    default_folder = ensure_patterns_dir()
    source_folder = _resolve_source_folder(default_folder)

    if not source_folder.exists() or not source_folder.is_dir():
        print(f"Folder not found: {source_folder}")
        return

    target_rows = _read_positive_int("Target rows (RR): ")
    target_cols = _read_positive_int("Target cols (CC): ")

    destination_name = f"pattern_{_format_dimension(target_rows)}_{_format_dimension(target_cols)}"
    destination_folder = source_folder.parent / destination_name
    destination_folder.mkdir(exist_ok=True)

    source_images = sorted(source_folder.glob("*.png"))
    if not source_images:
        print(f"No PNG images found in: {source_folder}")
        return

    created_count = 0
    skipped_count = 0
    cmap = ListedColormap(["white", "black"])

    for source_image in source_images:
        grid = _load_binary_png_any_size(source_image)
        if grid is None:
            print(f"Skipping {source_image.name}: unable to read as 2D PNG.")
            skipped_count += 1
            continue

        current_rows, current_cols = grid.shape
        if current_rows > target_rows or current_cols > target_cols:
            print(
                f"Skipping {source_image.name}: current size {current_rows}x{current_cols} is larger than target "
                f"{target_rows}x{target_cols}."
            )
            skipped_count += 1
            continue

        pad_rows = target_rows - current_rows
        pad_cols = target_cols - current_cols
        big_grid = np.pad(grid, ((0, pad_rows), (0, pad_cols)), mode="constant", constant_values=0)

        output_name = f"big_{source_image.name}"
        output_path = destination_folder / output_name
        plt.imsave(output_path, big_grid, cmap=cmap, vmin=0, vmax=1)
        created_count += 1
        print(f"Created: {output_path.name}")

    print(
        f"Upsize complete: {created_count} file(s) created in {destination_folder}. "
        f"Skipped: {skipped_count}."
    )


def run_view_folder_images_utility() -> None:
    """Display the first 8 PNG images from a selected folder in a new window."""
    default_folder = ensure_patterns_dir()
    source_folder = _resolve_folder_input(default_folder, "Folder to view")

    if not source_folder.exists() or not source_folder.is_dir():
        print(f"Folder not found: {source_folder}")
        return

    image_files = sorted(source_folder.glob("*.png"))[:8]
    if not image_files:
        print(f"No PNG images found in: {source_folder}")
        return

    images: list[np.ndarray] = []
    image_titles: list[str] = []
    dimensions: list[tuple[int, int]] = []

    for image_path in image_files:
        try:
            image = np.asarray(plt.imread(image_path))
            if image.ndim < 2:
                print(f"Skipping {image_path.name}: unsupported image dimensions.")
                continue
            images.append(image)
            image_titles.append(image_path.stem)
            dimensions.append((int(image.shape[0]), int(image.shape[1])))
        except OSError:
            print(f"Skipping {image_path.name}: unable to read image file.")

    if not images:
        print("No readable PNG images found to display.")
        return

    unique_dims = sorted(set(dimensions))
    if len(unique_dims) == 1:
        rows, cols = unique_dims[0]
        dims_text = f"{rows}x{cols}"
    else:
        dims_text = "mixed sizes"

    show_gallery_window(
        images,
        image_titles,
        suptitle=f"Folder Images: {source_folder.name} ({dims_text}, first {len(images)})",
        window_title=f"Folder Images - {source_folder.name}",
    )


def show_utilities_menu() -> None:
    print("\n=== Utilities ===")
    print("1. Upsize pattern images (pad right and bottom)")
    print("2. View folder images")
    print("3. Create 8 clean pixelated character images")
    for option in range(4, 6):
        print(f"{option}. Not implemented")
    print("0. Back")


def run_utilities_menu() -> None:
    """Run utilities submenu."""
    while True:
        show_utilities_menu()
        choice = input("Choose a utility option: ").strip()

        if choice == "1":
            run_upsize_patterns_utility()
        elif choice == "2":
            run_view_folder_images_utility()
        elif choice == "3":
            run_create_img_folder_utility()
        elif choice == "0":
            return
        elif choice.isdigit() and 4 <= int(choice) <= 5:
            print("This utility is not implemented yet.")
        else:
            print("Invalid choice. Please enter 0-5.")
