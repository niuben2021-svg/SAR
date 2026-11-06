"""Preserve official test partition; split only trainval by acquisition for validation."""
import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from data import find_partition_samples, parse_txt, sample_paths


def acquisition(stem):
    """Heuristic for RADARS-like GF3 filenames ending in two numeric tile indices."""
    parts = stem.rsplit("_", 2)
    if len(parts) == 3 and all(re.fullmatch(r"\d+", p) for p in parts[1:]):
        return parts[0]
    return stem


def main():
    p = argparse.ArgumentParser(description="Create train/val/test manifests from official partitions")
    p.add_argument("--data-root", required=True, help="Parent dataset folder containing trainval and test")
    p.add_argument("--out", default="split.json")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-ratio", type=float, default=0.15)
    args = p.parse_args()
    if not 0 <= args.val_ratio < 1:
        p.error("--val-ratio must be in [0,1)")
    root = Path(args.data_root)
    trainval = find_partition_samples(root, "trainval")
    test = find_partition_samples(root, "test")
    all_classes = Counter()
    classes_by_partition = {}
    for partition_name, samples in (("trainval", trainval), ("test", test)):
        counter = Counter()
        for rec in samples:
            _, txt = sample_paths(root, rec)
            counter.update(ann["label"] for ann in parse_txt(txt))
        classes_by_partition[partition_name] = dict(sorted(counter.items()))
        all_classes.update(counter)
    groups = defaultdict(list)
    for rec in trainval:
        groups[acquisition(rec["stem"])].append(rec)
    group_keys = sorted(groups)
    random.Random(args.seed).shuffle(group_keys)
    if args.val_ratio == 0 or len(group_keys) < 2:
        n_val = 0
        if args.val_ratio and len(group_keys) < 2:
            print("WARNING: only one trainval acquisition group; validation empty to prevent leakage.")
    else:
        n_val = min(len(group_keys) - 1, max(1, round(len(group_keys) * args.val_ratio)))
    val_groups = set(group_keys[:n_val])
    splits = {
        "train": sorted((r for k in group_keys if k not in val_groups for r in groups[k]), key=lambda r:r["image"]),
        "val": sorted((r for k in group_keys if k in val_groups for r in groups[k]), key=lambda r:r["image"]),
        "test": test,
    }
    test_groups = {acquisition(r["stem"]) for r in test}
    leakage = set(group_keys).intersection(test_groups)
    if leakage:
        print(f"WARNING: {len(leakage)} acquisition identifiers overlap official trainval/test! "
              f"Verify scene-level independence. Examples: {sorted(leakage)[:3]}")
    train_classes = set()
    for rec in splits["train"]:
        _, txt = sample_paths(root, rec)
        train_classes.update(a["label"] for a in parse_txt(txt))
    unseen = sorted(set(all_classes) - train_classes)
    if unseen:
        print(f"WARNING: classes without training examples: {unseen}. "
              "Reconsider group validation split or remove impossible classes.")
    record = {
        "format_version": 2,
        "data_root_at_creation": str(root.resolve()),
        "class_names": sorted(all_classes),
        "splits": splits,
        "seed": args.seed,
        "val_ratio": args.val_ratio,
        "official_test_untouched": True,
        "grouping": "trainval group = filename without final two numeric tile suffixes (when applicable)",
        "num_acquisition_groups_in_trainval": len(group_keys),
        "class_counts_in_original_partitions": classes_by_partition,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf8")
    print("All classes:", record["class_names"])
    print("Image counts:", {k:len(v) for k,v in splits.items()})
    print("Source object counts:", classes_by_partition)
    print("Saved:", args.out)


if __name__ == "__main__":
    main()
