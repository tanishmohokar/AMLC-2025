"""
Step 4: parse structured features out of catalog_content (CPU only).

Every catalog entry carries lines like:
    Value: 72.0
    Unit: Fl Oz
These become two features joined with the embeddings by LightGBM and CatBoost:
    value  - float (0.0 if missing)
    units  - normalised unit string, used as a categorical ("none" if missing)

Outputs: features.csv (train), features_test.csv (test)

Usage: python extract_features.py
"""
import re

import pandas as pd

from common import TEST_CSV, TEST_FEATURES_CSV, TRAIN_CSV, TRAIN_FEATURES_CSV, load_raw

VALUE_PATTERN = re.compile(r"Value:\s*([\d.]+)")
UNIT_PATTERN = re.compile(r"Unit:\s*(.*)")

# Collapses the spelling variants seen in the data onto one label per unit
UNIT_ALIASES = {
    "ounce": "ounce", "ounces": "ounce", "oz": "ounce",
    "fl oz": "fl oz", "fl. oz": "fl oz", "fl.oz": "fl oz", "fluid ounce": "fl oz", "fluid ounces": "fl oz",
    "count": "count", "ct": "count", "each": "count", "pack": "count",
    "pound": "pound", "pounds": "pound", "lb": "pound", "lbs": "pound",
    "gram": "gram", "grams": "gram", "gramm": "gram", "g": "gram",
    "kilogram": "kilogram", "kg": "kilogram",
    "millilitre": "milliliter", "milliliter": "milliliter", "ml": "milliliter",
    "liter": "liter", "liters": "liter", "litre": "liter", "l": "liter",
    "none": "none", "": "none",
}


def normalise_unit(unit):
    unit = re.sub(r"\s+", " ", unit.strip().lower())
    return UNIT_ALIASES.get(unit, unit)


def parse_catalog(text):
    value_match = VALUE_PATTERN.search(text)
    unit_match = UNIT_PATTERN.search(text)
    value = float(value_match.group(1)) if value_match else 0.0
    units = normalise_unit(unit_match.group(1)) if unit_match else "none"
    return value, units


def build_features(csv_path, out_path):
    df = load_raw(csv_path)
    parsed = df["catalog_content"].map(parse_catalog)
    features = pd.DataFrame({
        "sample_id": df["sample_id"],
        "value": [v for v, _ in parsed],
        "units": [u for _, u in parsed],
    })
    features.to_csv(out_path, index=False)
    print(f"Saved {len(features)} rows to {out_path}")
    print(features["units"].value_counts().head(10))


if __name__ == "__main__":
    build_features(TRAIN_CSV, TRAIN_FEATURES_CSV)
    build_features(TEST_CSV, TEST_FEATURES_CSV)
