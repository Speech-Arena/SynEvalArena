import warnings
import os
import argparse
import json
import pandas as pd
import torch
import tempfile
import pickle
import subprocess
import multiprocessing
import torch.multiprocessing as mp

from src.utils import write_summary
from src.asr import plot_phonemes
from src.speaker import compute_speaker_similarity
from src.asr import asr_inference
from src.mos import compute_wvmos, compute_speech_mos

warnings.filterwarnings('ignore')

def process_eval_wrapper(args):
    import pandas as pd

    flag, func, df, eval_flags, lang = args
    if eval_flags.get(flag, False):
        print(f"[Process] Running: {flag}")

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        torch.cuda.set_device(device)

        df_copy = df.copy()

        if flag == "asr_inference":
            result_df = func(df_copy, lang=lang)
        else:
            result_df = func(df_copy)

        if not isinstance(result_df, pd.DataFrame):
            raise TypeError(f"{flag} did not return a DataFrame")

        return result_df

    return df

def run_task_external_script(func_name: str, df: pd.DataFrame) -> pd.DataFrame:
    with tempfile.NamedTemporaryFile(delete=False) as in_file, tempfile.NamedTemporaryFile(delete=False) as out_file:
        input_path = in_file.name
        output_path = out_file.name

    with open(input_path, "wb") as f:
        pickle.dump(df, f)

    script_path = "run_utmos2_eval.py"
    subprocess.run(["python3", script_path, input_path, output_path], check=True)

    with open(output_path, "rb") as f:
        result_df = pickle.load(f)

    os.remove(input_path)
    os.remove(output_path)

    return result_df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_csv_path', required=True)
    parser.add_argument('--output_csv_path', required=True)
    parser.add_argument('--lang', default='english')
    parser.add_argument('--config_path', default='eval_config.json')
    parser.add_argument('--num_workers', type=int, default=multiprocessing.cpu_count())

    args = parser.parse_args()

    from src.metrics import (
        compute_pesq, compute_energy_js_divergence, compute_pitch_js_divergence,
        compute_pitch_correlation, compute_energy_correlation, compute_stoi, compute_mcd, compute_warpq,
        compute_final_pitch_error, compute_gross_pitch_error, compute_voicing_decision_error, compute_log_f0_rmse
    )

    torch.set_num_threads(torch.get_num_threads())
    mp.set_start_method('spawn', force=True)
    ctx = mp.get_context('spawn')

    with open(args.config_path, 'r') as f:
        eval_flags = json.load(f)

    df = pd.read_csv(args.input_csv_path)

    eval_steps = {
        "compute_wvmos": compute_wvmos,
        "compute_speech_mos": compute_speech_mos,
        "compute_utmos2_mos": None,  # special handling below
        "compute_speaker_similarity": compute_speaker_similarity,
        "asr_inference": asr_inference,
        "compute_pesq": compute_pesq,
        "compute_energy_js_divergence": compute_energy_js_divergence,
        "compute_pitch_js_divergence": compute_pitch_js_divergence,
        "compute_pitch_correlation": compute_pitch_correlation,
        "compute_energy_correlation": compute_energy_correlation,
        "compute_final_pitch_error": compute_final_pitch_error,
        "compute_gross_pitch_error": compute_gross_pitch_error,
        "compute_voicing_decision_error": compute_voicing_decision_error,
        "compute_log_f0_rmse": compute_log_f0_rmse,
        "compute_stoi": compute_stoi,
        "compute_mcd": compute_mcd,
        "compute_warpq": compute_warpq
    }

    unsafe_flags = {"compute_utmos2_mos"}
    safe_task_args = []
    unsafe_task_args = []

    for flag, func in eval_steps.items():
        if eval_flags.get(flag, False):
            task = (flag, func, df, eval_flags, args.lang)
            if flag in unsafe_flags:
                unsafe_task_args.append(task)
            else:
                safe_task_args.append(task)

    print(f"Running {len(safe_task_args)} multiprocessing-safe evaluations with {min(len(safe_task_args), args.num_workers)} workers...")
    print(f"Running {len(unsafe_task_args)} evaluations externally to avoid multiprocessing issues...")

    result_dfs = []

    if safe_task_args:
        with ctx.Pool(processes=min(len(safe_task_args), args.num_workers)) as pool:
            result_dfs.extend(pool.map(process_eval_wrapper, safe_task_args))

    for task in unsafe_task_args:
        flag, func, df, _, _ = task
        print(f"[Main] Running in subprocess: {flag}")
        if flag == "compute_utmos2_mos":
            result_dfs.append(run_task_external_script(flag, df))
        else:
            result_df = func(df)
            if not isinstance(result_df, pd.DataFrame):
                raise TypeError(f"{flag} did not return a DataFrame")
            result_dfs.append(result_df)

    for result_df in result_dfs:
        df = pd.concat([df, result_df.drop(columns=df.columns, errors='ignore')], axis=1)

    df.to_csv(args.output_csv_path, index=False, float_format='%.3f')

    try:
        write_summary(df, args.output_csv_path.replace('.csv', '.txt'))
    except AttributeError as e:
        print(f"Warning: Could not write full summary. Missing column? {e}")

    if eval_flags.get("plot_phonemes", False):
        print("Plotting phonemes")
        plot_phonemes(df, args.output_csv_path.replace('.csv', '.png'))

    for f in os.listdir('.'):
        if f.endswith('.wav') or f.endswith('.flac'):
            os.remove(f)

if __name__ == '__main__':
    main()
