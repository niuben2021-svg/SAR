"""SAR dataset IO: official dataset/{trainval,test}/{PNGImages,Annotations,...}.

Relative image/label paths in split.json prevent train and test stems from colliding.
No MAT, XML or keypoints are needed for the amplitude-image baseline.
"""
import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def parse_txt(path):
    annotations = []
    for line_no, line in enumerate(Path(path).read_text(encoding="utf-8-sig").splitlines(), 1):
        parts = line.split()
        if not parts:
            continue
        if len(parts) < 9:
            raise ValueError(f"{path}:{line_no}: expected eight coordinates and class, got {line!r}")
        try:
            p = np.asarray([float(x) for x in parts[:8]], dtype=np.float32).reshape(4, 2)
        except ValueError as exc:
            raise ValueError(f"{path}:{line_no}: invalid polygon") from exc
        if not np.isfinite(p).all():
            raise ValueError(f"{path}:{line_no}: nonfinite polygon")
        xyxy = [float(p[:, 0].min()), float(p[:, 1].min()),
                float(p[:, 0].max()), float(p[:, 1].max())]
        if xyxy[2] - xyxy[0] < 1 or xyxy[3] - xyxy[1] < 1:
            continue
        annotations.append({"box": xyxy, "label": parts[8]})
    return annotations


def load_gray(path):
    with Image.open(path) as img:
        return np.asarray(img.convert("L"), dtype=np.uint8).copy()


def to_tensor(gray):
    x = torch.from_numpy(np.ascontiguousarray(gray)).float().unsqueeze(0) / 255.0
    return x.expand(3, -1, -1).contiguous()


def sample_paths(root, record):
    """Resolve a sample record from split.json; never infer paths from a bare stem."""
    return Path(root) / record["image"], Path(root) / record["annotation"]


def find_partition_samples(root, partition):
    """Accept dataset/trainval and dataset/test, including extra wrapper directories.

    Example: dataset/SL-TRAINVAL/trainval/PNGImages is also accepted.
    Only directories with a parent literally named trainval or test are considered.
    """
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {root}")
    matches = sorted([p for p in root.rglob("PNGImages")
                      if p.is_dir() and p.parent.name.lower() == partition.lower()])
    if not matches:
        raise FileNotFoundError(
            f"Cannot locate {partition}/PNGImages below {root}. "
            f"Expected {root}/{partition}/PNGImages and {root}/{partition}/Annotations")
    if len(matches) != 1:
        raise ValueError(f"Multiple candidate {partition}/PNGImages folders: {matches}")
    image_dir = matches[0]
    annot_dir = image_dir.parent / "Annotations"
    if not annot_dir.is_dir():
        raise FileNotFoundError(f"Missing annotation directory: {annot_dir}")
    result = []
    missing = []
    # Restrict to files directly inside PNGImages, no repeated nested image copies.
    images = sorted(p for p in image_dir.iterdir()
                    if p.is_file() and p.suffix.lower() == ".png")
    for png in images:
        label = annot_dir / f"{png.stem}.txt"
        if not label.is_file():
            missing.append(png.name)
            continue
        result.append({
            "image": png.relative_to(root).as_posix(),
            "annotation": label.relative_to(root).as_posix(),
            "stem": png.stem,
        })
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} PNG files in {image_dir} lack a matching TXT in {annot_dir}. "
            f"First examples: {missing[:5]}")
    if not result:
        raise ValueError(f"No PNG+TXT pairs in {image_dir} / {annot_dir}")
    return result


def annotations_in_crop(annotations, labels_to_id, x0, y0, size, min_visible=0.35):
    boxes, labels = [], []
    for ann in annotations:
        x1, y1, x2, y2 = ann["box"]
        cx1 = max(x1, x0) - x0
        cy1 = max(y1, y0) - y0
        cx2 = min(x2, x0 + size) - x0
        cy2 = min(y2, y0 + size) - y0
        if cx2 <= cx1 or cy2 <= cy1:
            continue
        visible = ((cx2 - cx1) * (cy2 - cy1)
                   / max(1.0, (x2 - x1) * (y2 - y1)))
        if visible < min_visible or cx2 - cx1 < 2 or cy2 - cy1 < 2:
            continue
        boxes.append([cx1, cy1, cx2, cy2])
        labels.append(labels_to_id[ann["label"]])
    return {
        "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
        "labels": torch.tensor(labels, dtype=torch.int64),
    }


class TrainTiles(Dataset):
    def __init__(self, data_root, samples, class_names, tile=512,
                 tiles_per_image=8, positive_fraction=0.75):
        self.root = Path(data_root)
        self.samples = list(samples)
        self.labels_to_id = {name: i + 1 for i, name in enumerate(class_names)}
        self.tile = int(tile)
        self.tiles_per_image = int(tiles_per_image)
        self.positive_fraction = float(positive_fraction)
        if not self.samples:
            raise ValueError("Empty train split. See prepare.py output.")
        self.annotations = []
        for rec in self.samples:
            _, label_path = sample_paths(self.root, rec)
            anns = parse_txt(label_path)
            unknown = {a["label"] for a in anns} - self.labels_to_id.keys()
            if unknown:
                raise ValueError(f"Unknown classes {unknown} in {label_path}")
            self.annotations.append(anns)

    def __len__(self):
        return len(self.samples) * self.tiles_per_image

    def __getitem__(self, index):
        sample_idx = index % len(self.samples)
        image_path, _ = sample_paths(self.root, self.samples[sample_idx])
        gray = load_gray(image_path)
        h, w = gray.shape
        side = self.tile
        if h < side or w < side:
            raise ValueError(f"{image_path}: {w}x{h} smaller than --tile {side}")
        anns = self.annotations[sample_idx]
        if anns and random.random() < self.positive_fraction:
            obj = random.choice(anns)
            x1, y1, x2, y2 = obj["box"]
            cx = (x1 + x2) / 2 + random.uniform(-0.12, 0.12) * side
            cy = (y1 + y2) / 2 + random.uniform(-0.12, 0.12) * side
            x0 = max(0, min(w - side, round(cx - side / 2)))
            y0 = max(0, min(h - side, round(cy - side / 2)))
        else:
            x0 = random.randint(0, w - side)
            y0 = random.randint(0, h - side)
        target = annotations_in_crop(anns, self.labels_to_id, x0, y0, side)
        return to_tensor(gray[y0:y0 + side, x0:x0 + side]), target


def collate_fn(batch):
    return tuple(zip(*batch))


def read_split(path):
    obj = json.loads(Path(path).read_text(encoding="utf8"))
    if obj.get("format_version") != 2:
        raise ValueError("Old split.json layout. Rerun the updated prepare.py for dataset/trainval + dataset/test.")
    return obj["class_names"], obj["splits"]
