import evaluate
import numpy as np


def make_compute_metrics(id2label, entity=None):
    """Returns compute_metrics for token classification: seqeval over non -100 tokens.
    entity (e.g. "LOC"): precision/recall/f1 of that type only; accuracy stays over all tags."""
    metric = evaluate.load("seqeval")

    def compute_metrics_fn(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)

        true_labels = [
            [id2label[l] for l in label_row if l != -100] for label_row in labels
        ]
        true_preds = []
        for pred_row, label_row in zip(preds, labels):
            row = []
            for p, l in zip(pred_row, label_row):
                if l != -100:
                    row.append(id2label[int(p)])
            true_preds.append(row)

        results = metric.compute(predictions=true_preds, references=true_labels)
        scores = (
            results.get(entity, {"precision": 0.0, "recall": 0.0, "f1": 0.0})
            if entity
            else {k: results[f"overall_{k}"] for k in ("precision", "recall", "f1")}
        )
        return {
            "precision": scores["precision"],
            "recall": scores["recall"],
            "f1": scores["f1"],
            "accuracy": results["overall_accuracy"],
        }

    return compute_metrics_fn
