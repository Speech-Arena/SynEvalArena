import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import tqdm

# Must be set BEFORE importing transformers
#os.environ["TRANSFORMERS_NO_TORCHVISION"] = "1"

from voxprofile.src.model.emotion.wavlm_emotion import WavLMWrapper as WavLMWrapperEmotion 
from voxprofile.src.model.voice_quality.wavlm_voice_quality import WavLMWrapper as WavLMWrapperVoiceQuality
from voxprofile.src.model.accent.wavlm_accent import WavLMWrapper as WavLMWrapperAccent
from voxprofile.src.model.fluency.wavlm_fluency import WavLMWrapper as WavLMWrapperFluency
import torchaudio

class MultiWavLMEmbedder(nn.Module):
    """
    Loads four WavLMWrapper models and returns concatenated embeddings.
    Input must be (B, T) at 16kHz, <= 15s.
    """

    def __init__(self, device="cuda"):
        super().__init__()
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.models = {
            "accent": WavLMWrapperAccent.from_pretrained("tiantiaf/wavlm-large-broader-accent"),
            "emotion":WavLMWrapperEmotion.from_pretrained("tiantiaf/wavlm-large-categorical-emotion"),
            "speechflow": WavLMWrapperAccent.from_pretrained("tiantiaf/wavlm-large-speech-flow"),
            "voicequality":WavLMWrapperEmotion.from_pretrained("tiantiaf/wavlm-large-voice-quality")
        }
        self.cos = nn.CosineSimilarity(dim=1)
        for m in self.models:
            self.models[m].to(self.device)
            self.models[m].eval()

    def read_wav(self, path):
        
        wav, sr = torchaudio.load(path)
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
            wav = resampler(wav)
        return wav


    def compute_cosine_emotion(self, wav1, wav2, label=None):
        cos_distances = []
        for w1, w2 in tqdm.tqdm(zip(wav1, wav2)):
            w1 = self.read_wav(w1)
            w2 = self.read_wav(w2)
            emb1 = self.compute_emb(w1, label)
            emb2 = self.compute_emb(w2, label)
            dist = self.cos(emb1, emb2)
            #print(emb1.shape, emb2.shape, dist.shape)
            #print(dist.cpu().numpy().item())
            cos_distances.append(dist.cpu().numpy().item())
        
        return cos_distances

    @torch.no_grad()
    def compute_emb(self, batch_audio, label):
        """
        Args:
            batch_audio: (B, T) tensor of float32 audio at 16kHz (max 15s)

        Returns:
            torch.Tensor: (B, D_total) concatenated embeddings
        """
        max_len = 15 * 16000
        if batch_audio.size(1) > max_len:
            batch_audio = batch_audio[:, :max_len]

        batch_audio = batch_audio.to(self.device)
        emb = self.models[label](batch_audio, return_feature=True)
        return emb


def run_profile_process(df):
    print(f"[INFO] Running Accent Emotion Similarity with model:")
    profile_infer = MultiWavLMEmbedder()
    accent_sims = profile_infer.compute_cosine_emotion(df[0]["wavpath"], df[1]["wavpath"], label="accent")
    emotion_sims = profile_infer.compute_cosine_emotion(df[0]["wavpath"], df[1]["wavpath"], label="emotion")
    speechflow_sims = profile_infer.compute_cosine_emotion(df[0]["wavpath"], df[1]["wavpath"], label="speechflow")
    voicequality_sims = profile_infer.compute_cosine_emotion(df[0]["wavpath"], df[1]["wavpath"], label="voicequality")
    return pd.DataFrame([{
        "accent cosine similarity": accent_sims,
        "emotion_cosine_similarity": emotion_sims,
        "avg_accent_cosine_sim": np.mean(accent_sims),
        "avg_emotion_cosine_sim": np.mean(emotion_sims),
        "speechflow cosine similarity": speechflow_sims,
        "voicequality_cosine_similarity": voicequality_sims,
        "avg_speechflow_cosine_sim": np.mean(speechflow_sims),
        "avg_voicequality_cosine_sim": np.mean(voicequality_sims)
    }])

def run_profile_evaluation(df):
    outdf = pd.DataFrame()
    outdf = run_profile_process(df)
    return outdf

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_path', default='pfstar_asrbn_asv_ecapa512.csv', required=False)
    parser.add_argument('--config_path', default='/configs/privacy_config_simple.json', required=False)
    parser.add_argument('--language', required=False)
    parser.add_argument('--protocol_path', default='ASRBN_Emovdb_vp_protocol.xlsx', required=False)
    parser.add_argument('--model', required=False)

    args = parser.parse_args()

    import json 

    with open(args.config_path) as f:
        h = json.load(f)
    #df = get_df_sheet(args.protocol_path, args.orig_dir, args.anon_dir)    

    df = [pd.read_excel(args.protocol_path, sheet_name=i) for i in range(5)]
    
    outdf = run_profile_evaluation(df)
    outdf.to_csv(args.out_path, index=False, float_format='%.4f')
