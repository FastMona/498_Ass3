import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from multiprocessing import get_context
from pathlib import Path
import re
import numpy as np
import sys

from terminal_out import install_terminal_output_logger
from folder_prefs import prompt_for_folder


PATTERNS_DIR = Path(__file__).resolve().parent / "patterns"
GRID_ROWS = 12
GRID_COLS = 10


def _set_window_title(fig, window_title: str | None) -> None:
    """Set matplotlib window title when backend supports it."""
    manager = getattr(fig.canvas, "manager", None)
    if manager is None or not window_title:
        return
    try:
        manager.set_window_title(window_title)
    except Exception:
        return


def _show_grid_window_process(grid: np.ndarray, title: str, window_title: str | None) -> None:
    """Render one binary grid in a dedicated child process."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    cmap = ListedColormap(["white", "black"])
    fig, ax = plt.subplots(figsize=(12, 6))
    _set_window_title(fig, window_title)
    ax.imshow(grid, cmap=cmap, vmin=0, vmax=1, interpolation="none")
    ax.set_title(title)
    ax.set_xticks(range(grid.shape[1]))
    ax.set_yticks(range(grid.shape[0]))
    ax.set_xticks([i - 0.5 for i in range(grid.shape[1] + 1)], minor=True)
    ax.set_yticks([i - 0.5 for i in range(grid.shape[0] + 1)], minor=True)
    ax.grid(which="minor", color="lightgray", linewidth=0.5)
    ax.tick_params(which="minor", bottom=False, left=False)
    plt.tight_layout()
    plt.show()


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


def show_grid_window(grid: np.ndarray, title: str, window_title: str | None = None) -> None:
    """Start a process that shows a single grid window."""
    context = get_context("spawn")
    process = context.Process(
        target=_show_grid_window_process,
        args=(np.asarray(grid), title, window_title),
        daemon=True,
    )
    process.start()


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


def read_binary_row(row_index: int, total_rows: int, total_cols: int) -> list[int]:
    """Read one row of binary characters ('0' or '1') for requested dimensions."""
    while True:
        row = input(f"Row {row_index + 1}/{total_rows} ({total_cols} chars of 0 or 1): ").strip()
        if len(row) != total_cols:
            print(f"Invalid input: row must be exactly {total_cols} characters.")
            continue
        if any(ch not in {"0", "1"} for ch in row):
            print("Invalid input: only '0' and '1' are allowed.")
            continue
        return [int(ch) for ch in row]


def read_replacement_row(line_number: int, total_cols: int) -> list[int]:
    """Read replacement data for one selected line in edit mode."""
    while True:
        row = input(f"New data for line {line_number} ({total_cols} chars of 0 or 1): ").strip()
        if len(row) != total_cols:
            print(f"Invalid input: row must be exactly {total_cols} characters.")
            continue
        if any(ch not in {"0", "1"} for ch in row):
            print("Invalid input: only '0' and '1' are allowed.")
            continue
        return [int(ch) for ch in row]


def read_binary_row_or_repeat(
    row_index: int,
    total_rows: int,
    total_cols: int,
    previous_row: list[int] | None,
) -> list[int]:
    """Read one row, allowing Enter to repeat previous row when available."""
    while True:
        prompt = f"Row {row_index + 1}/{total_rows} ({total_cols} chars of 0 or 1, Enter to repeat last): "
        row = input(prompt).strip()
        if row == "":
            if previous_row is None:
                print("Invalid input: first row cannot be empty.")
                continue
            repeated_row = "".join(str(value) for value in previous_row)
            sys.stdout.write("\x1b[1A\r\x1b[2K")
            sys.stdout.write(f"{prompt}{repeated_row}\n")
            sys.stdout.flush()
            return previous_row.copy()

        if len(row) != total_cols:
            print(f"Invalid input: row must be exactly {total_cols} characters.")
            continue
        if any(ch not in {"0", "1"} for ch in row):
            print("Invalid input: only '0' and '1' are allowed.")
            continue
        return [int(ch) for ch in row]


def create_image_grid(total_rows: int, total_cols: int) -> list[list[int]]:
    """Read a binary grid from user input for requested dimensions."""
    print(f"\nCreate {total_cols}x{total_rows} image")
    print(f"Enter {total_rows} rows. Each row must contain exactly {total_cols} characters of 0 or 1.")
    print(f"Tip: press Enter on rows 2-{total_rows} to repeat the previous row.")

    grid: list[list[int]] = []
    for i in range(total_rows):
        previous_row = grid[-1] if grid else None
        grid.append(read_binary_row_or_repeat(i, total_rows, total_cols, previous_row))
    return grid


def read_line_number(total_rows: int) -> int:
    """Read one line number between 1 and total_rows for edit mode."""
    while True:
        value = input(f"Enter line number to edit (1-{total_rows}): ").strip()
        if value.isdigit() and 1 <= int(value) <= total_rows:
            return int(value)
        print(f"Invalid input: line number must be 1-{total_rows}.")


def read_dimension_value(prompt: str, default_value: int) -> int:
    """Read a positive dimension integer with a default value."""
    while True:
        raw = input(f"{prompt} [{default_value}]: ").strip()
        if raw == "":
            return default_value
        if raw.isdigit() and int(raw) > 0:
            return int(raw)
        print("Invalid input: enter a positive whole number.")


def ensure_patterns_dir() -> Path:
    """Ensure patterns output folder exists and return its path."""
    PATTERNS_DIR.mkdir(exist_ok=True)
    return PATTERNS_DIR


def get_default_pattern_name(folder: Path) -> str:
    """Return next default pattern name as pattern_XX."""
    max_number = 0
    for item in folder.iterdir():
        if not item.is_file():
            continue
        match = re.fullmatch(r"pattern_(\d+)", item.stem)
        if match:
            max_number = max(max_number, int(match.group(1)))

    next_number = max_number + 1
    if next_number < 100:
        return f"pattern_{next_number:02d}"
    return f"pattern_{next_number}"


def get_output_path(folder: Path) -> Path:
    """Prompt user with default name and return selected output path."""
    default_name = get_default_pattern_name(folder)
    chosen_name = input(f"Save file name [{default_name}]: ").strip() or default_name
    output_path = folder / chosen_name
    if output_path.suffix == "":
        output_path = output_path.with_suffix(".png")
    return output_path


def get_output_path_with_default(folder: Path, default_name: str) -> Path:
    """Prompt user with provided default name and return selected output path."""
    chosen_name = input(f"Save file name [{default_name}]: ").strip() or default_name
    output_path = folder / chosen_name
    if output_path.suffix == "":
        output_path = output_path.with_suffix(".png")
    return output_path


def normalize_pattern_base_name(raw_name: str) -> str:
    """Normalize user file name input to a base pattern name without extension."""
    name = raw_name.strip()
    if not name:
        return ""
    return Path(name).stem


def pattern_image_path(folder: Path, base_name: str) -> Path:
    """Return the .png path that stores the pattern image."""
    return folder / f"{base_name}.png"


def load_pattern_image(image_path: Path) -> list[list[int]] | None:
    """Load editable pattern rows from a binary PNG file of any size."""
    if not image_path.exists():
        return None

    try:
        data = plt.imread(image_path)
    except OSError:
        return None

    if data.ndim == 3:
        data = data[..., :3].mean(axis=2)

    if data.ndim != 2:
        return None

    grid_array = (np.asarray(data) < 0.5).astype(int)
    grid = grid_array.tolist()
    return grid


def display_image_grid(grid: list[list[int]], save_path: Path | None = None) -> None:
    """Display the binary grid where 0=white and 1=black."""
    cmap = ListedColormap(["white", "black"])
    grid_rows = len(grid)
    grid_cols = len(grid[0]) if grid_rows > 0 else 0

    image_title = f"Binary Image ({grid_rows} X {grid_cols}) (0=white, 1=black)"

    if save_path is not None:
        grid_array = np.asarray(grid, dtype=np.uint8)
        plt.imsave(save_path, grid_array, cmap=cmap, vmin=0, vmax=1)
        print(f"Saved image to: {save_path}")

    show_grid_window(np.asarray(grid, dtype=np.uint8), title=image_title, window_title="Pattern Image")


def display_recent_patterns(folder: Path) -> None:
    """Display a 2x4 figure of the 8 most recently saved patterns."""
    recent_files = sorted(
        folder.glob("*.png"),
        key=lambda file_path: file_path.stat().st_mtime,
        reverse=True,
    )[:8]

    if not recent_files:
        print(f"No PNG patterns found in: {folder}")
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
        size_text = f"{rows} X {cols}"
    else:
        size_text = "mixed sizes"

    show_gallery_window(
        images,
        image_titles,
        suptitle=f"Original Patterns ({size_text})",
        window_title="Most Recent Patterns",
    )


def run_create_image() -> None:
    """Create or edit and then display one GRID_ROWS x GRID_COLS binary image."""
    default_folder = ensure_patterns_dir()
    output_folder = prompt_for_folder("create_img.folder", "Folder for patterns", default_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    edit_name_raw = input("File name to create/edit (or '0' to view last 8 patterns): ").strip()

    if edit_name_raw == "0":
        display_recent_patterns(output_folder)
        return

    image: list[list[int]]
    output_path: Path

    base_name = normalize_pattern_base_name(edit_name_raw)
    if not base_name:
        base_name = get_default_pattern_name(output_folder)

    edit_image_path = pattern_image_path(output_folder, base_name)
    loaded_image = load_pattern_image(edit_image_path)

    if loaded_image is not None:
        image = loaded_image
        total_rows = len(image)
        total_cols = len(image[0]) if total_rows > 0 else 0
        if total_rows <= 0 or total_cols <= 0:
            print(f"Cannot edit '{base_name}': image has invalid dimensions.")
            return
        line_number = read_line_number(total_rows)
        print("Edit mode: only one line is edited per run.")
        print(f"Current row {line_number}: {''.join(str(v) for v in image[line_number - 1])}")
        replacement = read_replacement_row(line_number, total_cols)
        image[line_number - 1] = replacement
    else:
        rows = read_dimension_value("Pattern rows", GRID_ROWS)
        cols = read_dimension_value("Pattern cols", GRID_COLS)
        image = create_image_grid(rows, cols)

    output_path = get_output_path_with_default(output_folder, base_name)

    display_image_grid(image, save_path=output_path)


if __name__ == "__main__":
    install_terminal_output_logger()
    run_create_image()
