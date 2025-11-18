import numpy as np
import librosa
import pyworld as pw
import soundfile as sf
from scipy.stats import pearsonr
from scipy.spatial import distance
from librosa.sequence import dtw
from pystoi import stoi
from pesq import pesq
from speechpy.processing import cmvnw
from skimage.util.shape import view_as_windows
from src.mcd import Calculate_MCD
from pyvad import vad
import pandas as pd

N_FFT, HOP_LENGTH, WIN_LENGTH = 1024, 256, 1024


def stft(y):
    # Ensure y is a numpy array
    if not isinstance(y, np.ndarray):
        y = np.array(y)
    return librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH)

def pitch(x, sr=16000):
    return pw.dio(x.astype(np.float64), fs=sr, frame_period=HOP_LENGTH / sr * 1000)[0]

def energy(x, sr=16000):
    return np.sqrt(np.sum(librosa.magphase(stft(x))[0] ** 2, axis=0)).squeeze()

def align_with_dtw(x, y):
    # Ensure inputs are 1D arrays
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("Inputs x and y must be 1D arrays.")
    
    #print(f"Input shapes: x: {x.shape}, y: {y.shape}")
    
    # Perform DTW alignment
    _, wp = dtw(x, y, subseq=True, metric='euclidean')
    wp = np.array(wp[::-1])  # Reverse the warping path
    
    # Align x and y based on the warping path
    x_aligned = x[wp[:, 0]]
    y_aligned = y[wp[:, 1]]
    
    #print(f"Aligned shapes: x: {x_aligned.shape}, y: {y_aligned.shape}")
    return x_aligned, y_aligned

def js_divergence(x, y):
    return distance.jensenshannon(x, y)

def correlation(x, y):
    #print(x.shape, y.shape)
    return pearsonr(x, y)[0]

#def compute_stoi(df, col_name='STOI'):
#    def compute(row):
#        ref, sr = sf.read(row.ref_audio_path)
#        syn, sr = sf.read(row.syn_audio_path)
#        ref, syn = align_with_dtw(ref, syn)
#        return stoi(ref, syn, sr, extended=False)
#    df[col_name] = df.apply(compute, axis=1)
#    return df

def compute_stoi(df, col_name='STOI'):
    def compute(row):
        ref, sr_ref = sf.read(row.ref_audio_path)
        syn, sr_syn = sf.read(row.syn_audio_path)

        # Ensure same sampling rate
        if sr_ref != sr_syn:
            raise ValueError(f"Sampling rates differ: {sr_ref} vs {sr_syn}")

        # Convert to mono if stereo
        if ref.ndim > 1:
            ref = np.mean(ref, axis=1)
        if syn.ndim > 1:
            syn = np.mean(syn, axis=1)

        # --- Zero padding alignment ---
        max_len = max(len(ref), len(syn))

        if len(ref) < max_len:
            ref = np.pad(ref, (0, max_len - len(ref)))
        if len(syn) < max_len:
            syn = np.pad(syn, (0, max_len - len(syn)))

        # Compute STOI
        return stoi(ref, syn, sr_ref, extended=False)

    df[col_name] = df.apply(compute, axis=1)
    return df






def compute_pesq(df, col_name='PESQ'):
    def compute(row):
        ref, sr = sf.read(row.ref_audio_path)
        syn, sr = sf.read(row.syn_audio_path)
        return pesq(sr, ref, syn, 'wb')
    df[col_name] = df.apply(compute, axis=1)
    return df

def compute_mcd(df, col_name='MCD', mode="dtw"):
    def compute(row):
        return Calculate_MCD(MCD_mode=mode).calculate_mcd(row.ref_audio_path, row.syn_audio_path)    
    df[col_name] = df.apply(compute, axis=1)
    return df

def compute_warpq(df, col_name='warpq'):
    # Constants
    sr = 16000
    n_mfcc = 12
    fmax = 5000
    patch_size = 0.4
    sigma = np.array([[1, 1], [3, 2], [1, 3]])
    win_length = int(0.032 * sr)
    hop_length = int(0.004 * sr)
    n_fft = 2 * win_length
    lifter = 3
    cols = int(patch_size / (hop_length / sr))
    step = cols // 2

    def extract_mfcc(y, sr):
        """Extract MFCC features with normalization."""
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, fmax=fmax,
                                    n_fft=n_fft, win_length=win_length,
                                    hop_length=hop_length, lifter=lifter)
        return cmvnw(mfcc.T, win_size=201, variance_normalization=True).T

    def compute(row):
        """Compute the WARP-Q metric for a single row."""
        try:
            # Read audio files
            ref, sr_ref = sf.read(row.ref_audio_path)
            syn, sr_syn = sf.read(row.syn_audio_path)

            # Resample if necessary
            ref = librosa.resample(ref, sr_ref, sr) if sr_ref != sr else ref
            syn = librosa.resample(syn, sr_syn, sr) if sr_syn != sr else syn

            # Clip audio to [-1, 1]
            ref, syn = np.clip(ref, -1, 1), np.clip(syn, -1, 1)

            # Apply VAD
            ref_vad = ref[vad(ref, sr, fs_vad=sr, hop_length=30, vad_mode=0) == 1]
            syn_vad = syn[vad(syn, sr, fs_vad=sr, hop_length=30, vad_mode=0) == 1]

            # Extract MFCC features
            ref_mfcc = extract_mfcc(ref_vad, sr)
            syn_mfcc = extract_mfcc(syn_vad, sr)

            # Create overlapping patches
            patches = view_as_windows(syn_mfcc, (ref_mfcc.shape[0], cols), step=step)[0]

            # Compute DTW for each patch
            dtw_scores = [
                dtw(patch, ref_mfcc, metric='euclidean', step_sizes_sigma=sigma,
                    weights_mul=[1] * 3, band_rad=0.25, subseq=True)[0][-1, -1] / patch.shape[1]
                for patch in patches
            ]

            # Return the median DTW score
            return np.median(dtw_scores)

        except Exception as e:
            # Handle errors gracefully
            print(f"Error processing row {row}: {e}")
            return np.nan

    # Apply the compute function to each row
    df[col_name] = df.apply(compute, axis=1)
    return df

def voicing_decision(f0):
    return (f0 > 0).astype(np.int32)

def voicing_decision_error(f0_ref, f0_syn):
    v_ref = voicing_decision(f0_ref)
    v_syn = voicing_decision(f0_syn)
    return 100 * np.mean(v_ref != v_syn)

def gross_pitch_error(f0_ref, f0_syn, threshold=0.2):
    v_ref = voicing_decision(f0_ref)
    v_syn = voicing_decision(f0_syn)
    voiced = (v_ref == 1) & (v_syn == 1)
    if np.sum(voiced) == 0:
        return 0
    rel_err = np.abs(f0_ref[voiced] - f0_syn[voiced]) / f0_ref[voiced]
    return 100 * np.mean(rel_err > threshold)

def fine_pitch_error(f0_ref, f0_syn):
    v_ref = voicing_decision(f0_ref)
    v_syn = voicing_decision(f0_syn)
    voiced = (v_ref == 1) & (v_syn == 1)
    if np.sum(voiced) == 0:
        return 0
    rel_err = np.abs(f0_ref[voiced] - f0_syn[voiced]) / f0_ref[voiced]
    return 100 * np.mean(rel_err[rel_err <= 0.2])

def log_f0_rmse(f0_ref, f0_syn):
    """Root mean squared error between log-F0 of voiced frames."""
    v_ref = voicing_decision(f0_ref)
    v_syn = voicing_decision(f0_syn)
    voiced = (v_ref == 1) & (v_syn == 1)
    if np.sum(voiced) == 0:
        return 0
    return np.sqrt(np.mean((np.log(f0_ref[voiced]) - np.log(f0_syn[voiced])) ** 2))

def compute_stat_metric(df, feature_fn, metric_fn, col_name, sr=16000):
    def compute(row):
        syn, _ = sf.read(row.syn_audio_path)
        ref, _ = sf.read(row.ref_audio_path)
        #print(syn.shape, ref.shape)
        return metric_fn(*align_with_dtw(feature_fn(syn, sr), feature_fn(ref, sr)))
    df[col_name] = df.apply(compute, axis=1)
    return df

def compute_pitch_js_divergence(df):
    return compute_stat_metric(df, pitch, js_divergence, 'pitch_js_divergence')

def compute_energy_js_divergence(df):
    return compute_stat_metric(df, energy, js_divergence, 'energy_js_divergence')

def compute_pitch_correlation(df):
    return compute_stat_metric(df, pitch, correlation, 'pitch_correlation')

def compute_energy_correlation(df):
    return compute_stat_metric(df, energy, correlation, 'energy_correlation')

def compute_final_pitch_error(df):
    return compute_stat_metric(df, pitch, fine_pitch_error, 'fine_pitch_error')

def compute_gross_pitch_error(df):
    return compute_stat_metric(df, pitch, gross_pitch_error, 'gross_pitch_error')

def compute_voicing_decision_error(df):
    return compute_stat_metric(df, pitch, voicing_decision_error, 'voicing_decision_error')

def compute_log_f0_rmse(df):
    return compute_stat_metric(df, pitch, log_f0_rmse, 'log_f0_rmse')