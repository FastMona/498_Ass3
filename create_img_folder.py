from pathlib import Path
from io import BytesIO

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np


_WORKSPACE_DIR = Path(__file__).resolve().parent
_INVALID_FILENAME_CHARS = set('\\/:*?"<>|')
_DIGIT_TEMPLATES_12X10: dict[str, tuple[str, ...]] = {
    "0": (
        "0011111100",
        "0110000110",
        "1100000011",
        "1100000011",
        "1100000011",
        "1100000011",
        "1100000011",
        "1100000011",
        "0110000110",
        "0011111100",
        "0011111100",
        "0000000000",
    ),
    "1": (
        "0001100000",
        "0011100000",
        "0111100000",
        "0001100000",
        "0001100000",
        "0001100000",
        "0001100000",
        "0001100000",
        "0111111000",
        "0111111000",
        "0000000000",
        "0000000000",
    ),
    "2": (
        "0011111100",
        "0110000110",
        "0000000110",
        "0000001100",
        "0000011000",
        "0000110000",
        "0001100000",
        "0011000000",
        "0111111110",
        "0111111110",
        "0000000000",
        "0000000000",
    ),
    "3": (
        "0011111100",
        "0110000110",
        "0000000110",
        "0000011100",
        "0000011100",
        "0000000110",
        "0000000110",
        "1100000110",
        "0110001100",
        "0011111000",
        "0000000000",
        "0000000000",
    ),
    "4": (
        "0000011000",
        "0000111000",
        "0001111000",
        "0011011000",
        "0110011000",
        "1100011000",
        "1111111110",
        "0000011000",
        "0000011000",
        "0000011000",
        "0000000000",
        "0000000000",
    ),
    "5": (
        "0111111110",
        "0110000000",
        "0110000000",
        "0111111100",
        "0000000110",
        "0000000110",
        "0000000110",
        "0000000110",
        "0110000110",
        "0011111100",
        "0000000000",
        "0000000000",
    ),
    "6": (
        "0011111100",
        "0110000110",
        "0110000000",
        "0110000000",
        "0111111100",
        "0110000110",
        "0110000110",
        "0110000110",
        "0110000110",
        "0011111100",
        "0000000000",
        "0000000000",
    ),
    "7": (
        "0111111110",
        "0000000110",
        "0000001100",
        "0000011000",
        "0000110000",
        "0001100000",
        "0011000000",
        "0011000000",
        "0011000000",
        "0011000000",
        "0000000000",
        "0000000000",
    ),
    "8": (
        "0011111100",
        "0110000110",
        "0110000110",
        "0110000110",
        "0011111100",
        "0110000110",
        "0110000110",
        "0110000110",
        "0110000110",
        "0011111100",
        "0000000000",
        "0000000000",
    ),
    "9": (
        "0011111100",
        "0110000110",
        "0110000110",
        "0110000110",
        "0011111110",
        "0000000110",
        "0000000110",
        "0000000110",
        "0110000110",
        "0011111100",
        "0000000000",
        "0000000000",
    ),
}


def _read_non_empty_text(prompt: str, default_value: str | None = None) -> str:
    """Read non-empty input text, optionally using a default."""
    while True:
        value = input(prompt).strip()
        if value:
            return value
        if default_value is not None:
            return default_value
        print("Invalid input: value cannot be empty.")


def _read_positive_int(prompt: str) -> int:
    """Read a positive integer value from terminal input."""
    while True:
        raw = input(prompt).strip()
        if raw.isdigit() and int(raw) > 0:
            return int(raw)
        print("Invalid input: enter a positive whole number.")


def _read_exactly_eight_chars() -> str:
    """Read exactly 8 characters as the source alphabet for generation."""
    while True:
        value = input("Enter exactly 8 characters (as one string): ")
        if len(value) == 8:
            return value
        print("Invalid input: please enter exactly 8 characters.")


def _resolve_output_folder() -> Path:
    """Prompt for folder name/path and return ensured output folder path."""
    folder_input = _read_non_empty_text("Folder name for output [patterns]: ", default_value="patterns")
    candidate = Path(folder_input)
    if not candidate.is_absolute():
        candidate = _WORKSPACE_DIR / candidate
    candidate.mkdir(parents=True, exist_ok=True)
    return candidate


def _resize_nearest(binary_grid: np.ndarray, new_rows: int, new_cols: int) -> np.ndarray:
    """Resize a binary grid with nearest-neighbor sampling."""
    source_rows, source_cols = binary_grid.shape
    row_indices = np.linspace(0, source_rows - 1, new_rows).round().astype(int)
    col_indices = np.linspace(0, source_cols - 1, new_cols).round().astype(int)
    return binary_grid[row_indices][:, col_indices]


def _template_to_grid(template_rows: tuple[str, ...]) -> np.ndarray:
    """Convert a tuple of binary text rows into a uint8 numpy grid."""
    return np.asarray([[1 if ch == "1" else 0 for ch in row] for row in template_rows], dtype=np.uint8)


def _try_load_template_character_grid(character: str, rows: int, cols: int) -> np.ndarray | None:
    """Load canonical template grid for known characters, resizing when needed."""
    template_rows = _DIGIT_TEMPLATES_12X10.get(character)
    if template_rows is None:
        return None

    base_grid = _template_to_grid(template_rows)
    if tuple(base_grid.shape) == (rows, cols):
        return base_grid
    return _resize_nearest(base_grid, rows, cols)


def _render_character_binary_grid(character: str, rows: int, cols: int) -> np.ndarray:
    """Render one character to a centered binary grid of requested size."""
    figure = plt.figure(figsize=(1, 1), dpi=128)
    axis = figure.add_axes((0.0, 0.0, 1.0, 1.0))
    figure.patch.set_facecolor("white")
    axis.set_facecolor("white")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")

    axis.text(
        0.5,
        0.5,
        character,
        ha="center",
        va="center",
        fontsize=92,
        fontweight="bold",
        family="DejaVu Sans Mono",
        color="black",
    )

    buffer = BytesIO()
    figure.savefig(buffer, format="png", dpi=128, facecolor="white", edgecolor="white")
    plt.close(figure)
    buffer.seek(0)
    rgba = np.asarray(plt.imread(buffer))

    grayscale = rgba[..., :3].mean(axis=2)
    binary = (grayscale < 0.8).astype(np.uint8)

    occupied = np.argwhere(binary == 1)
    if occupied.size == 0:
        return np.zeros((rows, cols), dtype=np.uint8)

    row_min, col_min = occupied.min(axis=0)
    row_max, col_max = occupied.max(axis=0)
    cropped = binary[row_min : row_max + 1, col_min : col_max + 1]

    scale = min(rows / cropped.shape[0], cols / cropped.shape[1])
    scaled_rows = max(1, int(round(cropped.shape[0] * scale)))
    scaled_cols = max(1, int(round(cropped.shape[1] * scale)))
    resized = _resize_nearest(cropped, scaled_rows, scaled_cols)

    canvas = np.zeros((rows, cols), dtype=np.uint8)
    start_row = (rows - scaled_rows) // 2
    start_col = (cols - scaled_cols) // 2
    canvas[start_row : start_row + scaled_rows, start_col : start_col + scaled_cols] = resized
    return canvas


def _character_token(character: str) -> str:
    """Return a safe token for filenames while preserving simple characters."""
    if character.isalnum():
        return character
    if character == " ":
        return "space"
    if character in _INVALID_FILENAME_CHARS or character.isspace():
        return f"u{ord(character):04X}"
    return character


def run_create_img_folder_utility() -> None:
    """Create 8 clean pixelated character PNGs in a selected folder."""
    print("\n=== Create 8 Clean Pixelated Character Images ===")
    output_folder = _resolve_output_folder()
    rows = _read_positive_int("Pattern rows: ")
    cols = _read_positive_int("Pattern cols: ")
    character_text = _read_exactly_eight_chars()

    cmap = ListedColormap(["white", "black"])
    token_counts: dict[str, int] = {}

    for character in character_text:
        token_base = _character_token(character)
        token_counts[token_base] = token_counts.get(token_base, 0) + 1
        token = token_base
        if token_counts[token_base] > 1:
            token = f"{token_base}_{token_counts[token_base]}"

        image_name = f"pattern_{token}.png"
        image_path = output_folder / image_name
        template_grid = _try_load_template_character_grid(character, rows, cols)
        if template_grid is not None:
            grid = template_grid
        else:
            grid = _render_character_binary_grid(character, rows, cols)
        plt.imsave(image_path, grid, cmap=cmap, vmin=0, vmax=1)
        print(f"Created: {image_name}")

    print(f"Created 8 image files in: {output_folder}")


if __name__ == "__main__":
    run_create_img_folder_utility()
