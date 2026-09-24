"""Upload the Gradio app to its Hugging Face Space.

The Space gets space/app.py, space/requirements.txt, space/README.md and the
radio_imaging package, laid out the way the Space expects. Log in first with
`hf auth login` (or the older `huggingface-cli login`).

Usage, from the repo root:

    python space/deploy.py Faisal87/radio-imaging-audio-generator
    python space/deploy.py Faisal87/radio-imaging-audio-generator --stage-only some/folder
"""
import argparse
import shutil
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def stage(folder):
    """Copy the Space's files into `folder`, in the Space's layout."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    for name in ("app.py", "requirements.txt", "README.md"):
        shutil.copy(ROOT / "space" / name, folder / name)
    shutil.copytree(ROOT / "radio_imaging", folder / "radio_imaging",
                    ignore=shutil.ignore_patterns("__pycache__"), dirs_exist_ok=True)
    return folder


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("space_id", help="for example Faisal87/radio-imaging-audio-generator")
    parser.add_argument("--stage-only", metavar="FOLDER", help="only lay the files out in FOLDER, for a local test")
    args = parser.parse_args()

    if args.stage_only:
        print(f"Staged in {stage(args.stage_only)}")
        return

    from huggingface_hub import HfApi

    with tempfile.TemporaryDirectory() as folder:
        HfApi().upload_folder(folder_path=str(stage(folder)), repo_id=args.space_id, repo_type="space",
                              commit_message="Deploy from the GitHub fork")
    print(f"Uploaded. Open https://huggingface.co/spaces/{args.space_id}")


if __name__ == "__main__":
    main()
