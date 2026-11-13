import evaluate

from core.predict import tag_ids


def score_tags(preds, labels, id2label, entity=None):
    """seqeval over non -100 positions of predicted tag ids [N, T] vs label ids [N, T].
    entity (e.g. "LOC"): precision/recall/f1 of that type only; accuracy stays over all tags."""
    metric = evaluate.load("seqeval")
    true_labels, true_preds = [], []
    for pred_row, label_row in zip(preds, labels):
        keep = label_row != -100
        true_labels.append([id2label[int(l)] for l in label_row[keep]])
        true_preds.append([id2label[int(p)] for p in pred_row[keep]])

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


def make_compute_metrics(id2label, entity=None):
    """compute_metrics for Trainer: tag_ids tags (LOC threshold) scored by score_tags."""
    label2id = {label: i for i, label in id2label.items()}

    def compute_metrics_fn(eval_pred):
        logits, labels = eval_pred
        return score_tags(tag_ids(logits, label2id).numpy(), labels, id2label, entity)

    return compute_metrics_fn
