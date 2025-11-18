# -*- coding: utf-8 -*-
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.wvmos import get_wvmos
import torch
import librosa
from tqdm import tqdm
import utmosv2
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import soundfile as sf
import numpy as np
# ---------- Shared Device ----------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------- Main Evaluation Function ----------
def run_voicemos_evaluation(outdf, df):
    models = ["speechmos"]

    def eval_one_model(model_name, outdf, df):
        mos_model = MOSModel(model_name, device=DEVICE)
        orig_mos = mos_model.predict(df[0]["wavpath"])
        anon_mos = mos_model.predict(df[1]["wavpath"])

        print(model_name, len(df[0]["wavpath"]), len(orig_mos))
        print(model_name, len(df[1]["wavpath"]), len(anon_mos))

        #mos_dict = {
        #    f"orig_set_mos_{model_name}": orig_mos,
        #    f"anon_set_mos_{model_name}": anon_mos
        #}
        #mos_df = pd.DataFrame(mos_dict)

        avg_orig_set_mos = np.mean(orig_mos)
        avg_anon_set_mos = np.mean(anon_mos)
        avg_mos_dict = {
            f"avg_orig_set_mos_{model_name}": avg_orig_set_mos,
            f"avg_anon_set_mos_{model_name}": avg_anon_set_mos
        }
        avg_mos_df = pd.DataFrame(avg_mos_dict, index=[0])

        outdf = pd.concat([outdf, avg_mos_df], axis=1)

        return outdf

    for model_name in models:
        outdf = eval_one_model(model_name, outdf, df)

    return outdf


# ---------- Wrapper Model ----------
class MOSModel:
    def __init__(self, model_name, device=DEVICE):
        self.device = device
        self.model = {
            "speechmos": SpeechMOSModel(self.device),
        }.get(model_name)
        if self.model is None:
            raise ValueError(f"Unsupported model name: {model_name}")

    def predict(self, wav_paths):
        return self.model.predict_(wav_paths)


# ---------- Model Implementations ----------
class WVMOSModel:
    def __init__(self, device):
        self.model = get_wvmos()

    @torch.no_grad()
    def predict_(self, paths):
        print('wvmos')
        return [self.model.calculate_one(p) for p in tqdm(paths)]


class SpeechMOSModel:
    def __init__(self, device):
        self.device = device
        self.model = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True).to(device)
        self.min_len = 2 * 16000  # 2 seconds worth of samples

    def _process(self, path):
        wave, sr = self.load_and_resample(path)
        if len(wave) < self.min_len:
            # Pad with zeros at the end
            pad_length = self.min_len - len(wave)
            wave = np.pad(wave, (0, pad_length), mode="constant")
        wave_tensor = torch.from_numpy(wave).unsqueeze(0).to(self.device).float()
        return self.model(wave_tensor, sr).item()
    
    def load_and_resample(self, path, target_sr=16000):
        audio, sr = sf.read(path)
        if sr != target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        if audio.ndim > 1:
            audio = audio[:, 0]  # Take only the first channel if audio is multichannel
        return audio, target_sr

    @torch.no_grad()
    def predict_(self, paths):
        print('speechmos')
        return [self._process(p) for p in tqdm(paths)]


class UTMOS2Model:
    def __init__(self, device):
        self.device = device

        # Step 1: load model architecture
        self.model = utmosv2.create_model(pretrained=False)

        # Step 2: move to device safely using `to_empty`
        self.model.to_empty(device=self.device)

        # Step 3: load weights manually from cache
        model_path = "/idiap/temp/akulkarni/vp_work/eval_tool/sq/checkpoints/utmosv2.pt"
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)

    @torch.no_grad()
    def predict_(self, paths):
        print('utmosv2')
        return [
            self.model.predict(input_path=p, device=self.device, verbose=False)
            for p in tqdm(paths)
        ]



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_path', default='test_out.xlsx', required=False)
    parser.add_argument('--config_path', default='./configs/privacy_config.json', required=False)
    parser.add_argument('--language', default='English', required=False)
    parser.add_argument('--protocol_path', default='./filelists/LibriSpeech_Dev_m.xlsx', required=False)

    args = parser.parse_args()

    import json 

    with open(args.config_path) as f:
        h = json.load(f)

    
    df = [pd.read_excel(args.protocol_path, sheet_name=i) for i in range(5)]
    outdf = pd.DataFrame()
    outdf = run_voicemos_evaluation(outdf, df)
    outdf.to_csv(args.out_path, index=False, float_format='%.3f')