"""Create a reproducible image sample for the submission package.

Run from the project root:
python code/prepare_submission_samples.py --ddd dataset/driver_drowsiness_dataset --yawn dataset/yawn_eye_dataset_new --output dataset/samples --count 50 --seed 42
"""

import argparse
import csv
import hashlib
import random
import shutil
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def sha256_file(path):
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def sample_images(
    source_dir,
    output_dir,
    dataset_name,
    split_name,
    class_name,
    count,
    seed,
    manifest_rows,
):
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)

    if not source_dir.is_dir():
        raise FileNotFoundError(f"Directory not found: {source_dir}")

    images = sorted(
        path
        for path in source_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not images:
        raise RuntimeError(f"No valid images found in: {source_dir}")

    if len(images) < count:
        print(
            f"[WARNING] {source_dir} contains only {len(images)} images; "
            f"all images will be copied instead of {count}."
        )
        selected = images
    else:
        # Create a separate deterministic seed for each dataset/split/class.
        group_seed = f"{seed}:{dataset_name}:{split_name}:{class_name}"
        rng = random.Random(group_seed)
        selected = sorted(rng.sample(images, count))

    output_dir.mkdir(parents=True, exist_ok=True)

    for source_path in selected:
        destination_path = output_dir / source_path.name
        shutil.copy2(source_path, destination_path)

        manifest_rows.append(
            {
                "dataset": dataset_name,
                "split": split_name,
                "class": class_name,
                "original_filename": source_path.name,
                "sample_path": (
                    Path(dataset_name)
                    / (
                        Path(class_name)
                        if split_name == "all"
                        else Path(split_name) / class_name
                    )
                    / source_path.name
                ).as_posix(),
                "sha256": sha256_file(destination_path),
            }
        )

    print(
        f"[OK] {dataset_name}/{split_name}/{class_name}: "
        f"{len(selected)} images -> {output_dir}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Create a sample dataset for the CPV301 submission."
    )
    parser.add_argument(
        "--ddd",
        required=True,
        help="DDD directory containing drowsy/ and non_drowsy/.",
    )
    parser.add_argument(
        "--yawn",
        required=True,
        help="yawn_eye directory containing train/ and test/.",
    )
    parser.add_argument(
        "--output",
        default="../dataset/samples",
        help="Output directory. Default: ../dataset/samples",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=50,
        help="Number of images per class. Default: 50.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed. Default: 42.",
    )
    args = parser.parse_args()

    ddd_root = Path(args.ddd).resolve()
    yawn_root = Path(args.yawn).resolve()
    output_root = Path(args.output).resolve()

    manifest_rows = []

    # DDD: 50 images per class
    for class_name in ("drowsy", "non_drowsy"):
        sample_images(
            source_dir=ddd_root / class_name,
            output_dir=output_root / "DDD" / class_name,
            dataset_name="DDD",
            split_name="all",
            class_name=class_name,
            count=args.count,
            seed=args.seed,
            manifest_rows=manifest_rows,
        )

    # yawn_eye: 50 images per class per split
    for split_name in ("train", "test"):
        for class_name in ("yawn", "no_yawn"):
            sample_images(
                source_dir=yawn_root / split_name / class_name,
                output_dir=(
                    output_root
                    / "yawn_eye"
                    / split_name
                    / class_name
                ),
                dataset_name="yawn_eye",
                split_name=split_name,
                class_name=class_name,
                count=args.count,
                seed=args.seed,
                manifest_rows=manifest_rows,
            )

    manifest_path = output_root / "sample_manifest.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    with manifest_path.open(
        "w", newline="", encoding="utf-8-sig"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "dataset",
                "split",
                "class",
                "original_filename",
                "sample_path",
                "sha256",
            ],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"\nManifest created: {manifest_path}")
    print(f"Total images: {len(manifest_rows)}")


if __name__ == "__main__":
    main()
