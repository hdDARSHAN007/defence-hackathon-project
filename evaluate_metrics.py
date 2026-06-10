"""Offline evaluation utility for binary classification metrics.

Expected CSV columns:
- y_true: ground-truth labels
- y_pred: predicted labels

Example:
    python evaluate_metrics.py --input logs/evaluation_labels.csv \\
        --true-col y_true --pred-col y_pred --positive-label UNAUTHORIZED
"""

import argparse
import csv
import json
from pathlib import Path


def compute_binary_metrics(y_true, y_pred, positive_label):
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    if len(y_true) == 0:
        raise ValueError("No samples found")

    tp = fp = tn = fn = 0
    for t, p in zip(y_true, y_pred):
        t_pos = t == positive_label
        p_pos = p == positive_label

        if t_pos and p_pos:
            tp += 1
        elif (not t_pos) and p_pos:
            fp += 1
        elif t_pos and (not p_pos):
            fn += 1
        else:
            tn += 1

    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return {
        "total_samples": total,
        "positive_label": positive_label,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": [[tn, fp], [fn, tp]],
    }


def main():
    parser = argparse.ArgumentParser(description="Compute accuracy, precision, recall, F1, and confusion matrix")
    parser.add_argument("--input", required=True, help="Path to CSV containing ground truth and predictions")
    parser.add_argument("--true-col", default="y_true", help="Ground-truth column name")
    parser.add_argument("--pred-col", default="y_pred", help="Prediction column name")
    parser.add_argument("--positive-label", default="UNAUTHORIZED", help="Positive class label (default: 'UNAUTHORIZED')")
    parser.add_argument("--output-json", default="", help="Optional path to write metrics JSON")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    with input_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError("Input CSV has no header row")
        if args.true_col not in reader.fieldnames or args.pred_col not in reader.fieldnames:
            raise ValueError(
                f"CSV must contain '{args.true_col}' and '{args.pred_col}' columns. "
                f"Available columns: {list(reader.fieldnames)}"
            )
        rows = list(reader)

    if not rows:
        raise ValueError("Input CSV has no data rows")

    y_true = [str(row[args.true_col]) for row in rows]
    y_pred = [str(row[args.pred_col]) for row in rows]
    positive_label = str(args.positive_label)

    unique_labels = set(y_true) | set(y_pred)
    if len(unique_labels) > 2:
        raise ValueError(
            "This evaluator expects binary labels. "
            f"Found labels: {sorted(unique_labels)}"
        )

    metrics = compute_binary_metrics(y_true, y_pred, positive_label)

    print(f"Total samples: {metrics['total_samples']}")
    print(f"Positive label: {metrics['positive_label']}\n")
    print(f"Accuracy : {metrics['accuracy']:.4f} ({metrics['accuracy'] * 100:.2f}%)")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall   : {metrics['recall']:.4f}")
    print(f"F1 score : {metrics['f1_score']:.4f}\n")
    print("Confusion Matrix [[TN, FP], [FN, TP]]")
    print(metrics["confusion_matrix"])

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
