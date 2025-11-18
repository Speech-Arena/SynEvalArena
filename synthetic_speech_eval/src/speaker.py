from speechbrain.inference.speaker import SpeakerRecognition
import torch
import pandas as pd
from tqdm import tqdm
import numpy as np

def compute_speaker_similarity(df, threshold=0.25):
    """
    Computes speaker similarity by averaging embeddings of reference utterances for each speaker
    and comparing to synthesized utterance using cosine similarity.

    Adds a 'speaker_scores' column to the dataframe.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", run_opts={"device": device})

    # Precompute mean embeddings for each speaker
    spk_embeddings = {
        spk_id: torch.stack([
            model.encode_batch(model.load_audio(w).unsqueeze(0), None, normalize=False).cpu().squeeze()
            for w in df[df.spk_id == spk_id].ref_audio_path
        ]).mean(dim=0)
        for spk_id in df.spk_id.unique()
    }

    # Compute similarity
    df['speaker_scores'] = [
        model.similarity(
            spk_embeddings[spk_id],
            model.encode_batch(model.load_audio(syn).unsqueeze(0), None, normalize=False).cpu().squeeze()
        ).item()
        for spk_id, syn in tqdm(zip(df.spk_id, df.syn_audio_path), total=len(df))
    ]

    return df
