import torch

from core.datasets import first_subword_labels, split_words


def word_tags(model, tokenizer, words):
    """BIO tag per word: argmax at first subword of each word.
    One text per forward pass: no right padding, which bidirectional RNN would read."""
    enc = tokenizer(words, is_split_into_words=True, add_special_tokens=False)
    device = next(model.parameters()).device
    inp = torch.tensor([enc["input_ids"]], dtype=torch.long, device=device)  # [1, T]
    with torch.no_grad():
        pred_ids = model(input_ids=inp).logits.argmax(dim=-1).squeeze(0).tolist()
    keep = first_subword_labels(enc.word_ids(), [0] * len(words))
    return [model.config.id2label[p] for p, k in zip(pred_ids, keep) if k != -100]


def merge_spans(tags, spans, entity="LOC"):
    """Char spans [l, r) of entity: B starts one, I continues it; I without open entity starts one."""
    out = []
    for tag, (l, r) in zip(tags, spans):
        if tag == f"I-{entity}" and out and out[-1][2]:
            out[-1][1] = r
        elif tag in (f"B-{entity}", f"I-{entity}"):
            out.append([l, r, True])
        elif out:
            out[-1][2] = False  # close entity
    return [(l, r) for l, r, _ in out]


def predict_locations(model, tokenizer, texts):
    """Location strings per text, cut from text by predicted char spans."""
    model.eval()
    result = []
    for text in texts:
        words, spans = split_words(text)
        tags = word_tags(model, tokenizer, words) if words else []
        result.append([text[l:r] for l, r in merge_spans(tags, spans)])
    return result
