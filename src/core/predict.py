import torch

from core.datasets import first_subword_labels, split_words


def tag_ids(logits, label2id, loc_threshold=None):
    """Tag ids from logits [..., C]. loc_threshold None: argmax.
    Else LOC when P(B-LOC) + P(I-LOC) >= loc_threshold (B or I, whichever more likely),
    otherwise most likely non-LOC tag."""
    logits = torch.as_tensor(logits, dtype=torch.float)
    if loc_threshold is None:
        return logits.argmax(dim=-1)
    b, i = label2id["B-LOC"], label2id["I-LOC"]
    probs = logits.softmax(dim=-1)
    is_loc = probs[..., b] + probs[..., i] >= loc_threshold
    loc_tag = torch.where(probs[..., b] >= probs[..., i], b, i)
    other = probs.clone()
    other[..., [b, i]] = -1.0
    return torch.where(is_loc, loc_tag, other.argmax(dim=-1))


def word_tags(model, tokenizer, words, loc_threshold=None):
    """BIO tag per word, from first subword of each word.
    One text per forward pass: no right padding, which bidirectional RNN would read."""
    enc = tokenizer(words, is_split_into_words=True, add_special_tokens=False)
    device = next(model.parameters()).device
    inp = torch.tensor([enc["input_ids"]], dtype=torch.long, device=device)  # [1, T]
    with torch.no_grad():
        logits = model(input_ids=inp).logits.squeeze(0).cpu()
    pred_ids = tag_ids(logits, model.config.label2id, loc_threshold).tolist()
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


def predict_locations(model, tokenizer, texts, loc_threshold=None):
    """Location strings per text, cut from text by predicted char spans."""
    model.eval()
    result = []
    for text in texts:
        words, spans = split_words(text)
        tags = word_tags(model, tokenizer, words, loc_threshold) if words else []
        result.append([text[l:r] for l, r in merge_spans(tags, spans)])
    return result
