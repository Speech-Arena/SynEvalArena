# -*- coding: utf-8 -*-
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import librosa
import pyworld as pw
import soundfile as sf
from scipy.stats import pearsonr
from scipy.spatial import distance
from dtw import dtw
from tqdm import tqdm
import pandas as pd


def run_generic_evaluation(outdf, df, out_path):
    generic_infer = FeatMetrics()
    outdf["pitch_correlation"] = generic_infer.compute_batch_metric(
        df[0]["wavpath"], df[1]["wavpath"], generic_infer.pitch, generic_infer.correlation)
    outdf["energy_correlation"] = generic_infer.compute_batch_metric(
        df[0]["wavpath"], df[1]["wavpath"], generic_infer.energy, generic_infer.correlation)
#    outdf["pitch_JS"] = generic_infer.compute_batch_metric(
#        df[0]["wavpath"], df[1]["wavpath"], generic_infer.pitch, generic_infer.JS)
#    outdf["energy_JS"] = generic_infer.compute_batch_metric(
#        df[0]["wavpath"], df[1]["wavpath"], generic_infer.energy, generic_infer.JS)

    data = {
        "avg_pitch_correlation": np.mean(outdf["pitch_correlation"]),
        "avg_energy_correlation": np.mean(outdf["energy_correlation"]),
#        "avg_pitch_JS": np.mean(outdf["pitch_JS"]),
#        "avg_energy_JS": np.mean(outdf["energy_JS"])
    }
    avg_df = pd.DataFrame(data, index=[0])
    outdf = pd.concat([outdf, avg_df], axis=1)
    outdf.to_csv(out_path, index=False, float_format='%.3f')  # Write the data to the CSV file
    
    return outdf


class FeatMetrics:
    def __init__(self, N_FFT=1024, HOP_LENGTH=256, WIN_LENGTH=1024):
        self.N_FFT = N_FFT
        self.HOP_LENGTH = HOP_LENGTH
        self.WIN_LENGTH = WIN_LENGTH

    def stft(self, y):
        y = np.asarray(y)
        return librosa.stft(y, n_fft=self.N_FFT, hop_length=self.HOP_LENGTH, win_length=self.WIN_LENGTH)

    def pitch(self, x, sr=16000):

        p = pw.dio(x.astype(np.float64), fs=sr, frame_period=self.HOP_LENGTH / sr * 1000)[0]

        #print(p.shape)
        return p

    def energy(self, x, sr=16000):
        S, _ = librosa.magphase(self.stft(x))
        energy = np.sqrt(np.sum(S**2, axis=0))
        return energy

    def align_with_dtw(self, x, y):
        # Ensure inputs are 1D numpy arrays
        # DTW Alignment
        alignment = dtw(x, y, keep_internals=True)

        # Align signals based on the optimal warping path
        aligned_ref = x[alignment.index1]
        aligned_syn = y[alignment.index2]        

        return aligned_ref, aligned_syn


    def compute_single_metric(self, ref_audio, syn_audio, feature_fn):
        ref_audio, syn_audio = self.align_with_dtw(ref_audio, syn_audio)
        ref_feat = feature_fn(ref_audio)
        syn_feat = feature_fn(syn_audio)        
        return ref_feat, syn_feat

    def compute_batch_metric(self, ref_paths, syn_paths, feature_fn, metric_fn):
        def load_and_resample(path, target_sr=16000):
            audio, sr = sf.read(path)
            if sr != target_sr:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
            if audio.ndim > 1:
                audio = audio[:, 0]  # Take only the first channel if audio is multichannel
            return audio, target_sr
        
        metric_scores = []
        for ref_path, syn_path in tqdm(zip(ref_paths.to_list(), syn_paths.to_list())):
            try:
                ref_audio, sr_ref = load_and_resample(ref_path)
                syn_audio, sr_syn = load_and_resample(syn_path)
                
                assert sr_ref == sr_syn, f"Sample rates do not match: {sr_ref} vs {sr_syn}"
                
                refp = feature_fn(ref_audio, sr_ref)
                synp = feature_fn(syn_audio, sr_syn)
                
                refp, synp = self.align_with_dtw(refp, synp)
                metric_scores.append(metric_fn(refp, synp))
            except Exception as e:
                print(f"Error processing ref_path: {ref_path}, syn_path: {syn_path}")
                print(e)
        
        return metric_scores

    def correlation(self, refp, synp):
        #print(refp.shape, synp.shape)
        return pearsonr(refp, synp)[0]
    
    def JS(self, refp, synp):
        return distance.jensenshannon(refp, synp)
    


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_path', default='test_out.xlsx', required=False)
    parser.add_argument('--config_path', default='./configs/privacy_config_simple.json', required=False)
    parser.add_argument('--language', default='English', required=False)
    parser.add_argument('--protocol_path', default='./filelists/LibriSpeech_Dev_m.xlsx', required=False)

    args = parser.parse_args()

    import json 

    with open(args.config_path) as f:
        h = json.load(f)

    
    df = [pd.read_excel(args.protocol_path, sheet_name=i) for i in range(5)]
    outdf = pd.DataFrame()
    outdf = run_generic_evaluation(outdf, df, args.out_path)
    outdf.to_csv(args.out_path, index=False, float_format='%.3f')