from create_img import run_create_image
from create_img import ensure_patterns_dir
from noise import run_create_noisy_patterns
from cleanup import run_cleanup
from hopfield_nn import run_hopfield_training, get_trained_model_labels, run_pattern_recall, run_recall_error_report, run_repeat_recall_report
from terminal_out import install_terminal_output_logger
from pathlib import Path
import torch


def show_menu() -> None:
    """Print the main dashboard menu."""
    print("\n=== Image Dashboard ===")
    print("1. Patterns (10x12) Create/Save/Edit and View")
    print("2. Noisy Patterns Create/Save/Edit and View")
    print("3. Train Hopfield Networks (HOPS, HOPA)")
    print("4. Recall of Noisy Paterns and View")
    print("5. Error Report")
    print("6. Repeat Recall Report")
    for option in range(7, 9):
        print(f"{option}. Not implemented")
    print("9. Clean up")
    print("0. Exit")


def show_startup_status() -> None:
    """Print current environment and model/pattern status."""
    patterns_folder = ensure_patterns_dir()
    pattern_count = len(list(patterns_folder.glob("*.png")))

    workspace_dir = Path(__file__).resolve().parent
    noisy_patterns_dir = workspace_dir / "noisy_patterns"
    noisy_pattern_count = 0
    if noisy_patterns_dir.exists() and noisy_patterns_dir.is_dir():
        noisy_pattern_count += len(list(noisy_patterns_dir.glob("*.png")))

    patterns_present_text = "present" if pattern_count > 0 else "not present"
    cuda_available = torch.cuda.is_available()
    device_text = "GPU" if cuda_available else "CPU"

    trained_models = get_trained_model_labels()
    trained_text = " ".join(trained_models) if trained_models else "none"

    print("\n=== Status ===")
    print(f"Using: {device_text} Torch: {torch.__version__} CUDA: {cuda_available}")
    print(f"Patterns: {patterns_present_text}")
    print(f"Noisy Patterns: {noisy_pattern_count}")
    print(f"NN trained: {trained_text}")


def main() -> None:
    """Run the interactive dashboard loop until user exits."""
    while True:
        show_startup_status()
        show_menu()
        choice = input("Choose an option: ").strip()

        if choice == "1":
            run_create_image()
        elif choice == "2":
            run_create_noisy_patterns()
        elif choice == "3":
            run_hopfield_training()
        elif choice == "4":
            run_pattern_recall()
        elif choice == "5":
            run_recall_error_report()
        elif choice == "6":
            run_repeat_recall_report()
        elif choice == "9":
            run_cleanup()
        elif choice == "0":
            print("Goodbye.")
            break
        elif choice.isdigit() and 7 <= int(choice) <= 8:
            print("This option is not implemented yet.")
        else:
            print("Invalid choice. Please enter 0 or 1-9.")


if __name__ == "__main__":
    install_terminal_output_logger()
    main()
