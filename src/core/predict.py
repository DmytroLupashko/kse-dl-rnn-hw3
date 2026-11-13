import torch
from nltk import word_tokenize


def merge_bio_entities(tokens, bio_tags):
    entities, cur = [], None
    for tok, tag in zip(tokens, bio_tags):
        if tag.startswith("B-"):
            if cur:
                entities.append(cur)
            cur = {"text": tok, "label": tag[2:]}
        elif tag.startswith("I-") and cur and cur["label"] == tag[2:]:
            cur["text"] += " " + tok
        else:  # "O" or mismatched I-
            if cur:
                entities.append(cur)
                cur = None
    if cur:
        entities.append(cur)
    return entities


def infer(model, text, tokenizer, split=word_tokenize):
    device = next(model.parameters()).device
    tok = split(text)
    ids = tokenizer.convert_tokens_to_ids(tok)
    inp = torch.tensor([ids], dtype=torch.long, device=device)  # [1, T]

    model.eval()
    with torch.no_grad():
        out = model(input_ids=inp)  # TokenClassifierOutput
        logits = out.logits  # [1, T, C]
        pred_ids = logits.argmax(dim=-1).squeeze(0).tolist()

    bio_tags = [model.config.id2label[p] for p in pred_ids]
    # task scores locations only: ORG/PER entities are dropped
    return [e for e in merge_bio_entities(tok, bio_tags) if e["label"] == "LOC"]
