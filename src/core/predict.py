import torch

from core.datasets import first_subword_labels, split_words

# Best mean LOC F1 over 5 models on uk_geo valid (scripts/tune_loc_threshold.py); beat argmax on LB
LOC_THRESHOLD = 0.5


def tag_ids(logits, label2id, loc_threshold=LOC_THRESHOLD):
    """Tag ids from logits [..., C], via probs_tag_ids on their softmax."""
    probs = torch.as_tensor(logits, dtype=torch.float).softmax(dim=-1)
    return probs_tag_ids(probs, label2id, loc_threshold)


def probs_tag_ids(probs, label2id, loc_threshold=LOC_THRESHOLD):
    """Tag ids from tag probs [..., C]: LOC when P(B-LOC) + P(I-LOC) >= loc_threshold
    (B or I, whichever more likely), otherwise most likely non-LOC tag."""
    b, i = label2id["B-LOC"], label2id["I-LOC"]
    is_loc = probs[..., b] + probs[..., i] >= loc_threshold
    loc_tag = torch.where(probs[..., b] >= probs[..., i], b, i)
    other = probs.clone()
    other[..., [b, i]] = -1.0
    return torch.where(is_loc, loc_tag, other.argmax(dim=-1))


def word_probs(model, tokenizer, words):
    """Tag probs [W, C] per word, from first subword of each word.
    One text per forward pass: no right padding, which bidirectional RNN would read."""
    enc = tokenizer(words, is_split_into_words=True, add_special_tokens=False)
    device = next(model.parameters()).device
    inp = torch.tensor([enc["input_ids"]], dtype=torch.long, device=device)  # [1, T]
    with torch.no_grad():
        logits = model(input_ids=inp).logits.squeeze(0).cpu()
    keep = first_subword_labels(enc.word_ids(), [0] * len(words))
    return logits.softmax(dim=-1)[torch.tensor(keep) != -100]


def word_tags(members, words, loc_threshold=LOC_THRESHOLD, weights=None):
    """BIO tag per word from tag probs averaged over members: (model, tokenizer) pairs, same labels.
    weights: one per member, summing to 1; None: equal."""
    probs = torch.stack([word_probs(m, t, words) for m, t in members])
    w = torch.full((len(members),), 1 / len(members)) if weights is None else torch.tensor(weights)
    probs = (probs * w[:, None, None]).sum(dim=0)
    config = members[0][0].config
    pred_ids = probs_tag_ids(probs, config.label2id, loc_threshold).tolist()
    return [config.id2label[p] for p in pred_ids]


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


def predict_locations(members, texts, loc_threshold=LOC_THRESHOLD, weights=None):
    """Location strings per text, cut from text by char spans predicted by members (see word_tags)."""
    for model, _ in members:
        model.eval()
    result = []
    for text in texts:
        words, spans = split_words(text)
        tags = word_tags(members, words, loc_threshold, weights) if words else []
        result.append([text[l:r] for l, r in merge_spans(tags, spans)])
    return result
