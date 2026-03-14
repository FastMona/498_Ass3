from pathlib import Path
import shutil

from folder_prefs import PREFS_FILE


WORKSPACE_DIR = Path(__file__).resolve().parent
PATTERNS_DIR = WORKSPACE_DIR / "patterns"
NOISY_PATTERNS_DIR = WORKSPACE_DIR / "noisy_patterns"
TEMP_PATTERNS_DIR = WORKSPACE_DIR / "temp_patterns"
TERMINAL_OUT_PATH = WORKSPACE_DIR / "terminal_out.txt"


def confirm_yes_no_default_no(prompt: str) -> bool:
    """Return True only when user explicitly answers yes; default is no."""
    response = input(f"{prompt} [y/N]: ").strip().lower()
    return response in {"y", "yes"}


def confirm_delete_with_warning(prompt: str) -> bool:
    """Require two-step confirmation before deleting image files."""
    if not confirm_yes_no_default_no(prompt):
        return False

    print("*** --> ARE YOU SURE ? <-- ***")
    return confirm_yes_no_default_no("This will permanently delete files. Continue?")


def delete_png_files_in_folder(folder: Path) -> int:
    """Delete all PNG files in the given folder and return count deleted."""
    if not folder.exists() or not folder.is_dir():
        return 0

    deleted_count = 0
    for png_file in folder.glob("*.png"):
        png_file.unlink(missing_ok=True)
        deleted_count += 1
    return deleted_count


def delete_folder_contents(folder: Path) -> tuple[int, int]:
    """Delete all files/subfolders inside folder and return deleted file/dir counts."""
    if not folder.exists() or not folder.is_dir():
        return 0, 0

    deleted_files = 0
    deleted_dirs = 0

    for child in folder.iterdir():
        if child.is_file():
            child.unlink(missing_ok=True)
            deleted_files += 1
        elif child.is_dir():
            shutil.rmtree(child, ignore_errors=True)
            deleted_dirs += 1

    return deleted_files, deleted_dirs


def delete_other_temp_files(root_folder: Path) -> tuple[int, int]:
    """Delete common temp files and __pycache__ folders under root folder."""
    deleted_files = 0
    deleted_dirs = 0

    for pattern in ("**/*.pyc", "**/*.pyo", "**/*.tmp", "**/*.npz"):
        for file_path in root_folder.glob(pattern):
            if file_path.is_file():
                file_path.unlink(missing_ok=True)
                deleted_files += 1

    for pycache_dir in root_folder.glob("**/__pycache__"):
        if pycache_dir.is_dir():
            shutil.rmtree(pycache_dir, ignore_errors=True)
            deleted_dirs += 1

    return deleted_files, deleted_dirs


def read_keep_lines_default_100() -> int:
    """Read how many terminal log lines to keep; default is 100."""
    while True:
        value = input("Keep last number of lines in terminal_out.txt [100]: ").strip()
        if value == "":
            return 100
        if value.isdigit() and int(value) >= 0:
            return int(value)
        print("Invalid input: enter a non-negative integer.")


def trim_terminal_output_file(file_path: Path, keep_last_lines: int) -> tuple[int, int] | None:
    """Trim terminal output file to the last keep_last_lines and return before/after line counts."""
    if not file_path.exists() or not file_path.is_file():
        return None

    content = file_path.read_text(encoding="utf-8", errors="replace")
    lines = content.splitlines()
    before_count = len(lines)
    kept_lines = lines[-keep_last_lines:] if keep_last_lines > 0 else []
    output_text = "\n".join(kept_lines)
    if output_text:
        output_text += "\n"

    file_path.write_text(output_text, encoding="utf-8")
    return before_count, len(kept_lines)


def run_cleanup() -> None:
    """Run cleanup prompts for images, temp files, and terminal log trim."""
    print("\n=== Cleanup ===")

    if confirm_delete_with_warning("Delete all .png files in patterns?"):
        count = delete_png_files_in_folder(PATTERNS_DIR)
        print(f"Deleted {count} .png file(s) from patterns.")
    else:
        print("Skipped patterns cleanup.")

    if confirm_yes_no_default_no("Delete all .png files in noisy_patterns?"):
        count = delete_png_files_in_folder(NOISY_PATTERNS_DIR)
        print(f"Deleted {count} .png file(s) from noisy_patterns.")
    else:
        print("Skipped noisy pattern cleanup.")

    if confirm_yes_no_default_no("Delete animation files in temp_patterns?"):
        deleted_files, deleted_dirs = delete_folder_contents(TEMP_PATTERNS_DIR)
        print(
            f"Deleted animation outputs in temp_patterns: {deleted_files} file(s), "
            f"{deleted_dirs} folder(s)."
        )
    else:
        print("Skipped temp_patterns animation cleanup.")

    if confirm_yes_no_default_no("Delete other temp files?"):
        deleted_files, deleted_dirs = delete_other_temp_files(WORKSPACE_DIR)
        print(f"Deleted {deleted_files} temp file(s) and {deleted_dirs} __pycache__ folder(s).")

        if confirm_yes_no_default_no("Shorten terminal_out.txt now?"):
            keep_last_lines = read_keep_lines_default_100()
            trim_result = trim_terminal_output_file(TERMINAL_OUT_PATH, keep_last_lines)
            if trim_result is None:
                print("Skipped terminal_out.txt trim: file not found.")
            else:
                before_count, after_count = trim_result
                print(f"Trimmed terminal_out.txt lines: {before_count} -> {after_count} (kept last {keep_last_lines}).")
        else:
            print("Skipped terminal_out.txt trim.")
    else:
        print("Skipped other temp file cleanup.")

    if confirm_yes_no_default_no("Reset remembered folder defaults?"):
        if PREFS_FILE.exists() and PREFS_FILE.is_file():
            PREFS_FILE.unlink(missing_ok=True)
            print("Cleared remembered folder defaults.")
        else:
            print("No remembered folder defaults were found.")
    else:
        print("Kept remembered folder defaults.")
