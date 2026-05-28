from pathlib import Path
import runpy


def main():
    train_script = Path(__file__).resolve().parents[2] / "train.py"
    if not train_script.exists():
        raise FileNotFoundError(f"Could not find training entry point at {train_script}")

    runpy.run_path(str(train_script), run_name="__main__")


if __name__ == "__main__":
    main()
