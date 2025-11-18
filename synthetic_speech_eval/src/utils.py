import numpy as np
import librosa
import soundfile as sf
import pandas as pd

def read_audio(path, sr=16000):
    y, original_sr = sf.read(path)
    if original_sr != sr:
        y = librosa.resample(y, orig_sr=original_sr, target_sr=sr)
    return y, sr

def safe_clip(audio, min_val=-1.0, max_val=1.0):
    return np.clip(audio, min_val, max_val)

def normalize_audio(audio):
    return audio / (np.max(np.abs(audio)) + 1e-8)


def write_summary(df: pd.DataFrame, fname: str) -> None:
    """Write the mean of numeric, non-empty columns in a DataFrame to a file."""
    with open(fname, 'w') as f:
        for col in df.select_dtypes(include=np.number).columns:
            valid_data = df[col].dropna()
            if not valid_data.empty:
                f.write(f"{col}: {valid_data.mean():.3f}\n")