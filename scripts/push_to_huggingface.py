"""Uploads the trained model artifact + model card to a Hugging Face Hub
model repo. NOT run automatically by anything in this project -- publishing
to a personal HF account needs that account's own token and the owner's
explicit go-ahead, so this script is meant to be run by hand.

Usage:
  export HF_TOKEN=hf_...
  python scripts/push_to_huggingface.py --repo-id your-username/tell-deception-cues
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", required=True, help="e.g. your-username/tell-deception-cues")
    parser.add_argument(
        "--models-dir", default="models", help="Directory containing the trained artifacts."
    )
    parser.add_argument("--private", action="store_true", help="Create the repo as private.")
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("Set HF_TOKEN in your environment before running this script.")

    from huggingface_hub import HfApi

    models_dir = Path(args.models_dir)
    required = ["tell_classifier.skops", "model_card.md", "metadata.json"]
    missing = [f for f in required if not (models_dir / f).exists()]
    if missing:
        raise SystemExit(
            f"Missing {missing} in {models_dir}. Run scripts/train_and_save_model.py first."
        )

    api = HfApi(token=token)
    print(f"Creating (or reusing) repo {args.repo_id} ...")
    api.create_repo(repo_id=args.repo_id, repo_type="model", private=args.private, exist_ok=True)

    print("Uploading model card as README.md ...")
    api.upload_file(
        path_or_fileobj=str(models_dir / "model_card.md"),
        path_in_repo="README.md",
        repo_id=args.repo_id,
    )
    print("Uploading model artifact ...")
    api.upload_file(
        path_or_fileobj=str(models_dir / "tell_classifier.skops"),
        path_in_repo="tell_classifier.skops",
        repo_id=args.repo_id,
    )
    print("Uploading metadata ...")
    api.upload_file(
        path_or_fileobj=str(models_dir / "metadata.json"),
        path_in_repo="metadata.json",
        repo_id=args.repo_id,
    )
    print(f"Done: https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
