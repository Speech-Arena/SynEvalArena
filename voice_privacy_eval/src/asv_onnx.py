# -*- coding: utf-8 -*-
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from sklearn.metrics.pairwise import paired_distances
from pathlib import Path
from tqdm.auto import tqdm
from sklearn.metrics import roc_curve, roc_auc_score
from src.helpers import cllr, eer_from_ers, min_cllr
import numpy as np

import onnxruntime as ort
import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi

import concurrent as concurrent

from concurrent.futures import ThreadPoolExecutor

def run_asv_evaluation(outdf, df, h, model_name=None):
    """
    Run ASV evaluations for a single speaker verification model.
    Returns a DataFrame with EER, Cllr, minCllr, AUROC, scores, and labels for the model.
    """
    asv_model = ASV_model(h["asv_model_paths"][model_name], model_name)
    orig_results_df, lazy_anon_results_df, ignorant_anon_results_df = asv_model.eval_speaker_verification(df)
    # Combine the orig and anon metrics (columns) into outdf
    outdf = pd.concat([outdf, orig_results_df, lazy_anon_results_df, ignorant_anon_results_df], axis=1)
    return outdf

class ASV_model:
    def __init__(self, model_name, col_name):
        self.model_name = model_name
        self.session = self.init_onnx_session(model_name)
        self.col_name = col_name


    def init_onnx_session(self, onnx_path):
        so = ort.SessionOptions()
        so.inter_op_num_threads = 1
        so.intra_op_num_threads = 1
        session = ort.InferenceSession(onnx_path, sess_options=so, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        return session
    

    def compute_fbank(self, wav_path, 
                  num_mel_bins=80,
                  frame_length=25,
                  frame_shift=10,
                  dither=0.0):
        waveform, sample_rate = torchaudio.load(wav_path)
        if waveform.size(1) > 15 * sample_rate:
            waveform = waveform[:, :15 * sample_rate]
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            waveform = resampler(waveform)
        
        if waveform.dim() > 1:
            #print(waveform.shape)
            waveform = waveform[0, :]
            #print(waveform.shape)
        waveform = waveform * (1 << 15)
        #print(waveform.unsqueeze(0).shape)
        mat = kaldi.fbank(waveform.unsqueeze(0),
                        num_mel_bins=num_mel_bins,
                        frame_length=frame_length,
                        frame_shift=frame_shift,
                        dither=dither,
                        sample_frequency=sample_rate,
                        window_type='hamming',
                        use_energy=False)
        # CMN, without CVN
        mat = mat - torch.mean(mat, dim=0)
        return mat.unsqueeze(0).numpy()
    

    def get_speaker_embedding(self, path):
        feats = self.compute_fbank(path)
        embeddings = self.session.run(output_names=['embs'],
                             input_feed={'feats': feats})
        embeddings = np.array(embeddings[0])
        print(f"Speaker embedding shape: {embeddings.shape}")
        return embeddings

    def extract_enrollment_embeddings(self, spk2utt, wavscp):
        spk_embeddings = {}
        for spk, utts in spk2utt.items():
            embeddings = []
            for utt in utts:
                try:
                    embeddings.append(self.get_speaker_embedding(wavscp[utt]))
                except Exception as e:
                    raise Exception(f"Error getting speaker embedding for {wavscp[utt]}: {e}")
            spk_embeddings[spk] = np.mean(embeddings, axis=0)[0]

        #enroll_emb = np.nan_to_num(spk_embeddings, nan=0.0)
        return spk_embeddings

    def extract_trial_embeddings(self, wavscp):
        trial_embeddings = {}
        for utt, path in wavscp.items():
            try:
                trial_embeddings[utt] = self.get_speaker_embedding(path)[0]
            except Exception as e:
                raise Exception(f"Error getting speaker embedding for {path}: {e}")
        return trial_embeddings

    def evaluate_metrics(self, scores, labels, col_str_):
        tar_llr = scores[labels == 1]
        nontar_llr = scores[labels == 0]

        fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
        auroc = roc_auc_score(labels, scores)
        eer = eer_from_ers(fpr, tpr) * 100
        print(eer)

        return pd.DataFrame([{
            '%s_eer_%s'%(self.col_name, col_str_): eer,
            '%s_cllr_value_%s'%(self.col_name, col_str_): cllr(tar_llr, nontar_llr),
            '%s_min_cllr_value_%s'%(self.col_name, col_str_): min_cllr(tar_llr, nontar_llr),
            '%s_auroc_%s'%(self.col_name, col_str_): auroc, 
            '%s_labels_%s'%(self.col_name, col_str_): list(labels),
            '%s_scores_%s'%(self.col_name, col_str_): list(scores)
        }])

    def parse_eval_files(self, df):
        utt2spk = df.set_index('utt')['spk'].to_dict()
        wavscp = df.set_index('utt')['wavpath'].to_dict()
        spk2utt = df.groupby('spk')['utt'].apply(list).to_dict()
        return spk2utt, utt2spk, wavscp

    def compute_scores(self, enroll_embeddings, trial_pairs, trial_embeddings):
        enroll_emb = np.array([enroll_embeddings[spk] for spk, _, _ in trial_pairs])
        trial_emb = np.array([trial_embeddings[utt] for _, utt, _ in trial_pairs])
        labels = np.array([1 - int(label) for _, _, label in trial_pairs])

        scores = paired_distances(enroll_emb, trial_emb, metric='cosine')
        return scores, labels

    def eval_speaker_verification(self, sheets):
        trials = sheets[4][['spk', 'utt', 'label']].values.tolist() # trial sheet: 4

        def run_evaluation(enroll_sheet, dev_sheet, col_str_):
            enroll_spk2utt, _, enroll_wavscp = self.parse_eval_files(sheets[enroll_sheet])
            _, _, dev_wavscp = self.parse_eval_files(sheets[dev_sheet])

            #print(enroll_spk2utt)

            enroll_emb = self.extract_enrollment_embeddings(enroll_spk2utt, enroll_wavscp)
            trial_emb = self.extract_trial_embeddings({**dev_wavscp, **enroll_wavscp}) # if test samples are used as enrollment in trials

            scores, labels = self.compute_scores(enroll_emb, trials, trial_emb)
            return self.evaluate_metrics(scores, labels, col_str_)

        orig_results = run_evaluation(2,0, col_str_='orignal') #'Orignal_Enroll_set', 'Orignal_set'
        lazy_anon_results = run_evaluation(3,1, col_str_='lazy_informed_a2a')#'Annon_Enroll_set', 'Anon_set'
        ignorant_anon_results = run_evaluation(2, 1, col_str_='ingorant_o2a')#'Orignal_Enroll_set', 'Anon_set'
        
        return orig_results, lazy_anon_results, ignorant_anon_results





if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_path', default='globe_mcadmas_asv_ecapa512.csv', required=False)
    parser.add_argument('--config_path', default='./configs/privacy_config_simple.json', required=False)
    parser.add_argument('--language', default='English', required=False)
    parser.add_argument('--protocol_path', default='asrbn_vctk_protocols.xlsx', required=False)
    parser.add_argument('--model', default='asv_ecapa512', required=False)

    args = parser.parse_args()

    import json 

    with open(args.config_path) as f:
        h = json.load(f)
    #df = get_df_sheet(args.protocol_path, args.orig_dir, args.anon_dir)
    

    df = [pd.read_excel(args.protocol_path, sheet_name=i) for i in range(5)]
    
    outdf = pd.DataFrame()
    outdf = run_asv_evaluation(outdf, df, h, model_name=args.model)
    outdf.to_csv(args.out_path, index=False, float_format='%.3f')

