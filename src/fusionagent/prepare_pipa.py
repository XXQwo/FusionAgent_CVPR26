"""Prepare PIPA head/upper-body H5 files for FusionAgent.

This script deliberately does not synthesize gait or pretend the PIPA head box
is a full-body box. It uses the original PIPA geometry:
  head: annotated box
  upper body: 3 x head width and 3 x head height, with the head at top-centre
(the upper half of the 3w x 6h full-body rectangle used in the PIPA literature).

The input images must use the same coordinate system as the supplied PIPA boxes.
"""

import argparse
import csv
from pathlib import Path

import h5py
import numpy as np
from PIL import Image


def parse_line(line, expect_split=False):
    parts = line.strip().split()
    min_cols = 9 if expect_split else 8
    if len(parts) < min_cols:
        raise ValueError("Malformed PIPA metadata line: {}".format(line.rstrip()))
    album, photo = parts[0], parts[1]
    x, y, w, h = map(float, parts[2:6])
    pid, subset = int(parts[6]), int(parts[7])
    split = int(parts[8]) if expect_split else None
    return album, photo, x, y, w, h, pid, subset, split


def resolve_image(root, album, photo):
    root = Path(root)
    candidates = [
        root / f"f1_{album}_{photo}.jpg",
        root / f"{album}_{photo}.jpg",
        root / f"{photo}.jpg",
        root / album / f"{photo}.jpg",
        root / f"f1_{album}_{photo}.jpeg",
        root / f"{photo}.jpeg",
        root / f"{photo}.png",
    ]
    for p in candidates:
        if p.exists():
            return p
    # Last resort for reconstructed PIPA trees with arbitrary prefixes.
    hits = list(root.rglob(f"*{photo}*.jpg"))
    if len(hits) == 1:
        return hits[0]
    return None


def clip_box(box, width, height):
    x0, y0, x1, y1 = box
    x0 = max(0, min(width, int(round(x0))))
    y0 = max(0, min(height, int(round(y0))))
    x1 = max(0, min(width, int(round(x1))))
    y1 = max(0, min(height, int(round(y1))))
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def crop_regions(image, x, y, w, h):
    width, height = image.size
    head_box = clip_box((x, y, x + w, y + h), width, height)
    cx = x + 0.5 * w
    upper_box = clip_box((cx - 1.5 * w, y, cx + 1.5 * w, y + 3.0 * h), width, height)
    if head_box is None or upper_box is None:
        return None, None
    return image.crop(head_box), image.crop(upper_box)


def collect_rows(all_data, split_test):
    rows = []
    with open(all_data, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            album, photo, x, y, w, h, pid, subset, _ = parse_line(line, False)
            if subset == 0:
                rows.append((album, photo, x, y, w, h, pid, "train"))

    with open(split_test, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            album, photo, x, y, w, h, pid, _, split = parse_line(line, True)
            rows.append((album, photo, x, y, w, h, pid, f"test{split}"))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all-data", required=True, help="PIPA all_data.txt")
    parser.add_argument("--split-test", required=True, help="e.g. split_test_original.txt")
    parser.add_argument("--images", required=True, help="directory containing reconstructed/original PIPA images")
    parser.add_argument("--output", required=True, help="output data root; creates PIPA_FusionAgent/")
    parser.add_argument("--compression", default="gzip", choices=["gzip", "lzf", "none"])
    args = parser.parse_args()

    out_dir = Path(args.output) / "PIPA_FusionAgent"
    out_dir.mkdir(parents=True, exist_ok=True)
    head_path = out_dir / "pipa_head.h5"
    upper_path = out_dir / "pipa_upper.h5"
    manifest_path = out_dir / "manifest.csv"

    compression = None if args.compression == "none" else args.compression
    rows = collect_rows(args.all_data, args.split_test)
    missing = 0
    written = 0

    with h5py.File(head_path, "w") as head_h5, h5py.File(upper_path, "w") as upper_h5, open(
        manifest_path, "w", newline="", encoding="utf-8"
    ) as mf:
        fieldnames = [
            "key", "pid", "split", "album_id", "photo_id", "source_path",
            "head_x", "head_y", "head_w", "head_h",
        ]
        writer = csv.DictWriter(mf, fieldnames=fieldnames)
        writer.writeheader()

        for idx, (album, photo, x, y, w, h, pid, split_name) in enumerate(rows):
            img_path = resolve_image(args.images, album, photo)
            if img_path is None:
                missing += 1
                continue
            try:
                image = Image.open(img_path).convert("RGB")
            except Exception:
                missing += 1
                continue
            head, upper = crop_regions(image, x, y, w, h)
            if head is None or upper is None:
                missing += 1
                continue

            key = f"{split_name}/{idx:08d}"
            head_h5.create_dataset(key, data=np.asarray(head), compression=compression)
            upper_h5.create_dataset(key, data=np.asarray(upper), compression=compression)
            writer.writerow({
                "key": key,
                "pid": pid,
                "split": split_name,
                "album_id": album,
                "photo_id": photo,
                "source_path": str(img_path),
                "head_x": x, "head_y": y, "head_w": w, "head_h": h,
            })
            written += 1

    print(f"Prepared {written} PIPA instances; skipped {missing}.")
    print(f"Manifest: {manifest_path}")
    print(f"Head H5:  {head_path}")
    print(f"Upper H5: {upper_path}")
    if missing:
        print("WARNING: missing/skipped images change the benchmark population. Record this count in experiments.")


if __name__ == "__main__":
    main()
