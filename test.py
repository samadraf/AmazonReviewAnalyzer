
import pandas as pd
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent / "src"))
from src.preprocess import preprocess_text
from src.feature_extraction import extract_features


input_file = Path(__file__).resolve().parent / "fake reviews dataset.csv"
output_file = Path(__file__).resolve().parent / "processed-dataset.csv"

df = pd.read_csv(input_file)

df["cleaned_text"] = df["review"].apply(preprocess_text)

include_pos = True  # Set to True to include VERB, NOUN, ADV counts

df = extract_features(df, include_pos=include_pos)

df.to_csv(output_file, index=False)

print(f"✅ Processed dataset saved to: {output_file}")
