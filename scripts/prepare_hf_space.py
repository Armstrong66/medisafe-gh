"""Prepare a Hugging Face Space deployment bundle for the Gradio app."""

from __future__ import annotations

import argparse
import os
import shutil
import stat
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP_DIR = ROOT / "app"
DEFAULT_OUTPUT_DIR = ROOT / "dist" / "hf_space"
COMBINED_RESULTS = ROOT / "data" / "eval_outputs" / "combined" / "all_models_scored.jsonl"
SOURCE_DIRS = ("configs", "core", "models", "probes", "scorer", "scripts", "translation")
SOURCE_FILES = ("run_bilingual_eval.py",)


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    print(f"copied {src.relative_to(ROOT)} -> {dst.relative_to(ROOT)}")


def copy_tree(src: Path, dst: Path) -> None:
    if dst.exists():
        remove_tree(dst)
    shutil.copytree(
        src,
        dst,
        ignore=shutil.ignore_patterns(
            "__pycache__",
            "*.pyc",
            "*.pyo",
            "logs",
            "*.log",
            "run_codespaces_app.sh",
        ),
    )
    print(f"copied {src.relative_to(ROOT)} -> {dst.relative_to(ROOT)}")


def handle_remove_readonly(func, path, _exc_info) -> None:
    os.chmod(path, stat.S_IWRITE)
    func(path)


def remove_tree(path: Path) -> None:
    shutil.rmtree(path, onerror=handle_remove_readonly)


def prepare_space_bundle(output_dir: Path, include_results: bool = False) -> Path:
    if output_dir.exists():
        remove_tree(output_dir)
    output_dir.mkdir(parents=True)

    copy_file(APP_DIR / "app.py", output_dir / "app.py")
    copy_file(APP_DIR / "gmass_app.py", output_dir / "gmass_app.py")
    copy_file(APP_DIR / "spaces_README.md", output_dir / "README.md")
    copy_file(APP_DIR / "spaces_requirements.txt", output_dir / "requirements.txt")
    for source_dir in SOURCE_DIRS:
        copy_tree(ROOT / source_dir, output_dir / source_dir)
    for source_file in SOURCE_FILES:
        copy_file(ROOT / source_file, output_dir / source_file)
    (output_dir / ".gitignore").write_text(
        "\n".join(
            [
                "__pycache__/",
                "*.py[cod]",
                ".env",
                ".env.*",
                "logs/",
                "*.log",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"wrote {output_dir.relative_to(ROOT) / '.gitignore'}")

    if include_results:
        if not COMBINED_RESULTS.exists():
            raise FileNotFoundError(
                f"Cannot include benchmark results; missing {COMBINED_RESULTS.relative_to(ROOT)}"
            )
        copy_file(COMBINED_RESULTS, output_dir / COMBINED_RESULTS.relative_to(ROOT))

    print("")
    print(f"Space bundle ready at: {output_dir}")
    print("Next: copy or push that directory to your Hugging Face Space repository.")
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare the Hugging Face Space bundle.")
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Output directory for the generated Space bundle.",
    )
    parser.add_argument(
        "--include-results",
        action="store_true",
        help="Include precomputed combined benchmark results if available.",
    )
    args = parser.parse_args()

    prepare_space_bundle(Path(args.output_dir).resolve(), include_results=args.include_results)


if __name__ == "__main__":
    main()
