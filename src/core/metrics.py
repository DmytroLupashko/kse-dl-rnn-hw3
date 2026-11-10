import evaluate
import numpy as np
from sklearn.metrics import roc_auc_score


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits[:, 1]
    auc = roc_auc_score(labels, preds)
    return {"auc": auc}


def make_ner_compute_metrics(id2label):
    """Returns compute_metrics for token classification: seqeval over non -100 tokens."""
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
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }

    return compute_metrics_fn
