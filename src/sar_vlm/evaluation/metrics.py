"""Dependency-free classification, HBB detection and lexical/attribute metrics."""

import unicodedata
from collections import Counter

import numpy as np

from sar_vlm.evaluation.schema import Prediction, Protocol, Sample


def ratio(n: float, d: float) -> float:
    """Use zero for an undefined per-class precision/recall (documented convention)."""
    return float(n / d) if d else 0.0


def prf(tp: int, fp: int, fn: int) -> dict:
    """Count-based precision, recall, F1; never calibrated probabilities."""
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": ratio(tp, tp + fp),
        "recall": ratio(tp, tp + fn),
        "f1": ratio(2 * tp, 2 * tp + fp + fn),
    }


def classification(
    samples: list[Sample], predictions: dict[str, Prediction], classes: list[str]
) -> dict:
    """Image-level metrics; invalid/missing predictions remain in the denominator."""
    rows = [r for r in samples if r.label is not None]
    if not rows:
        return {"status": "NOT_ANNOTATED", "n": 0}
    counts = np.zeros((len(classes), len(classes) + 1), dtype=int)
    for row in rows:
        p = predictions[row.id]
        label = p.label if p.status == "ok" else None
        counts[
            classes.index(row.label), classes.index(label) if label in classes else len(classes)
        ] += 1
    per_class = {}
    for i, label in enumerate(classes):
        tp = int(counts[i, i])
        support = int(counts[i].sum())
        per_class[label] = {
            **prf(tp, int(counts[:, i].sum()) - tp, support - tp),
            "support": support,
        }
    supported = [v for v in per_class.values() if v["support"]]
    macro = {
        key: float(np.mean([v[key] for v in supported])) for key in ("precision", "recall", "f1")
    }
    return {
        "status": "OK",
        "n": len(rows),
        "accuracy": float(counts.trace() / len(rows)),
        "balanced_accuracy": macro["recall"],
        "macro_supported_classes": macro,
        "weighted_f1": sum(v["f1"] * v["support"] for v in supported) / len(rows),
        "invalid_or_abstained": int(counts[:, -1].sum()),
        "coverage": 1 - float(counts[:, -1].sum() / len(rows)),
        "per_class": per_class,
        "confusion_matrix": counts.tolist(),
        "row_labels": classes,
        "column_labels": classes + ["__invalid_or_abstain__"],
    }


def iou(a: tuple, b: tuple) -> float:
    """Continuous half-open original-pixel HBB intersection over union (no +1)."""
    area = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    return ratio(area, (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - area)


def match(ranked: list[tuple], ground: dict[str, list], threshold: float) -> tuple:
    """Greedy class-specific matching with one-to-one GT assignment, stable score ties."""
    used = {k: set() for k in ground}
    hits, overlaps = [], []
    for sample_id, obj in ranked:
        candidates = [
            (iou(obj.bbox_xyxy, gt.bbox_xyxy), j)
            for j, gt in enumerate(ground[sample_id])
            if j not in used[sample_id]
        ]
        best, index = max(candidates, default=(0.0, -1))
        hit = index >= 0 and best >= threshold
        if hit:
            used[sample_id].add(index)
            overlaps.append(best)
        hits.append(int(hit))
    return hits, overlaps


def ap101(hits: list[int], n_gt: int) -> float:
    """101-point interpolated AP, with full-dataset ranked predictions."""
    if not hits or not n_gt:
        return 0.0
    tp = np.cumsum(hits)
    precision = tp / np.arange(1, len(hits) + 1)
    recall = tp / n_gt
    return float(
        np.mean([max(precision[recall >= t], default=0.0) for t in np.linspace(0, 1, 101)])
    )


def detection(
    samples: list[Sample], predictions: dict[str, Prediction], protocol: Protocol
) -> dict:
    """HBB AP protocol without COCO crowd/ignore/area ranges; never label it COCOeval."""
    rows = [r for r in samples if r.objects is not None]
    if not rows:
        return {"status": "NOT_ANNOTATED", "n": 0}
    selected = {}
    missing_scores = 0
    truncated = 0
    for row in rows:
        p = predictions[row.id]
        objects = p.objects if p.status == "ok" else []
        missing_scores += sum(o.raw_score is None for o in objects)
        ranked = sorted(objects, key=lambda o: -(o.raw_score if o.raw_score is not None else 0))
        truncated += max(0, len(ranked) - protocol.max_detections)
        selected[row.id] = ranked[: protocol.max_detections]
    if missing_scores and protocol.score_threshold > 0:
        return {
            "status": "MISSING_SCORES",
            "n": len(rows),
            "reason": "positive score threshold cannot be applied to unscored detections",
        }
    per_class = {}
    aps, all_overlaps = [], []
    for label in protocol.classes:
        ground = {r.id: [o for o in r.objects if o.label == label] for r in rows}
        n_gt = sum(map(len, ground.values()))
        ranked = [(r.id, o) for r in rows for o in selected[r.id] if o.label == label]
        ranked.sort(key=lambda pair: -(pair[1].raw_score or 0))
        operating = [
            pair
            for pair in ranked
            if pair[1].raw_score is None or pair[1].raw_score >= protocol.score_threshold
        ]
        hits, overlaps = match(operating, ground, protocol.detection_iou)
        all_overlaps += overlaps
        values = (
            [ap101(match(ranked, ground, t / 100)[0], n_gt) for t in range(50, 100, 5)]
            if n_gt and not missing_scores
            else None
        )
        if values is not None:
            aps.append(values)
        per_class[label] = {
            **prf(sum(hits), len(hits) - sum(hits), n_gt - sum(hits)),
            "support": n_gt,
            "AP50": values[0] if values else None,
            "AP75": values[5] if values else None,
            "AP50_95": float(np.mean(values)) if values else None,
        }
    totals = {k: sum(v[k] for v in per_class.values()) for k in ("tp", "fp", "fn")}
    return {
        "status": "OK",
        "n": len(rows),
        **prf(**totals),
        "false_positives_per_image": totals["fp"] / len(rows),
        "miss_rate": ratio(totals["fn"], totals["tp"] + totals["fn"]),
        "matched_mean_iou": float(np.mean(all_overlaps)) if all_overlaps else None,
        "AP50": float(np.mean([x[0] for x in aps])) if aps else None,
        "AP75": float(np.mean([x[5] for x in aps])) if aps else None,
        "mAP50_95": float(np.mean(aps)) if aps else None,
        "ap_status": "OK" if aps else "NO_GT_OR_MISSING_RANKING_SCORES",
        "missing_scores": missing_scores,
        "truncated_detections": truncated,
        "per_class": per_class,
        "protocol": "HBB 101-point AP; IoU=.50:.05:.95; supported GT classes only; "
        "no crowd/ignore/area filtering; stable input-order score ties; "
        "AP uses all scores, P/R/F1 uses configured operating threshold",
    }


def characters(text: str) -> list[str]:
    """NFKC/casefold Unicode characters, excluding whitespace and punctuation."""
    return [
        c
        for c in unicodedata.normalize("NFKC", text).casefold()
        if not c.isspace() and not unicodedata.category(c).startswith("P")
    ]


def text_pair(prediction: str, reference: str) -> dict:
    """Character-unigram F1 and character ROUGE-L F1, not semantic factuality."""
    a, b = characters(prediction), characters(reference)
    overlap = sum((Counter(a) & Counter(b)).values())
    prev = [0] * (len(b) + 1)
    for x in a:
        curr = [0]
        for j, y in enumerate(b, 1):
            curr.append(prev[j - 1] + 1 if x == y else max(prev[j], curr[-1]))
        prev = curr
    return {
        "char_f1": ratio(2 * overlap, len(a) + len(b)),
        "rouge_l_char_f1": ratio(2 * prev[-1], len(a) + len(b)),
        "normalized_exact_match": float(bool(b) and a == b),
    }


def descriptions(samples: list[Sample], predictions: dict[str, Prediction]) -> dict:
    """Macro-average best-reference lexical similarity; failed outputs score zero."""
    rows = [r for r in samples if r.references is not None]
    if not rows:
        return {"status": "NOT_ANNOTATED", "n": 0}
    scores = []
    for row in rows:
        p = predictions[row.id]
        text = p.description if p.status == "ok" else ""
        options = [text_pair(text, ref) for ref in row.references]
        scores.append({key: max(v[key] for v in options) for key in options[0]})
    return {
        "status": "OK",
        "n": len(rows),
        **{k: float(np.mean([s[k] for s in scores])) for k in scores[0]},
        "protocol": "NFKC/casefold; ignore punctuation/space; best reference per metric; "
        "character-level lexical overlap does not verify physical correctness",
    }


def attributes(samples: list[Sample], predictions: dict[str, Prediction]) -> dict:
    """Closed-vocabulary attribute accuracy and contradictions on reviewed known slots."""
    rows = [r for r in samples if r.attributes is not None]
    if not rows:
        return {"status": "NOT_ANNOTATED", "n": 0}
    correct = wrong = omitted = unknown_claims = unknown_slots = 0
    for row in rows:
        p = predictions[row.id]
        output = p.attributes if p.status == "ok" else {}
        for key, truth in row.attributes.items():
            value = output.get(key)
            if truth is None:
                unknown_slots += 1
                unknown_claims += value is not None
            elif value is None:
                omitted += 1
            elif value == truth:
                correct += 1
            else:
                wrong += 1
    known = correct + wrong + omitted
    return {
        "status": "OK" if known else "NO_KNOWN_ATTRIBUTES",
        "n": len(rows),
        "known_slots": known,
        "correct": correct,
        "contradictions": wrong,
        "omitted_or_abstained": omitted,
        "accuracy_known": ratio(correct, known) if known else None,
        "coverage_known": ratio(correct + wrong, known) if known else None,
        "contradiction_rate_answered": ratio(wrong, correct + wrong) if correct + wrong else None,
        "assertion_micro": prf(correct, wrong, wrong + omitted),
        "unknown_reference_slots": unknown_slots,
        "claims_on_unknown": unknown_claims,
        "note": "Unknown reference slots cannot prove hallucination; only known contradictory "
        "values count as contradictions. Attributes are image-level, not object-linked.",
    }
