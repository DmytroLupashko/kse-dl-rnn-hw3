import pandas as pd

from core.predict import LOC_THRESHOLD, predict_locations
from helpers.config import SUBMISSIONS_DIR, TEST_PATH


def write_submission(model, tokenizer, name, loc_threshold=LOC_THRESHOLD):
    """Kaggle CSV (text_id, locations) for test.csv into SUBMISSIONS_DIR/<name>.csv; returns path."""
    test = pd.read_csv(TEST_PATH)
    locations = predict_locations(model, tokenizer, test["text"], loc_threshold)
    test["locations"] = [str(locs) for locs in locations]
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    out = SUBMISSIONS_DIR / f"{name}.csv"
    test[["text_id", "locations"]].to_csv(out, index=False)
    return out
