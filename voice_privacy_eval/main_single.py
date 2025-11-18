import os
import torch
import pandas as pd
import argparse
import json
import subprocess
import time
from itertools import islice
import openpyxl
import shutil
import sys
import random
import string
import concurrent.futures

def random_txt():
    return ''.join(random.choices(string.ascii_letters, k=4))

def validate_wav_paths(xlsx_file):
    required_order1 = ["Orignal_set", "Anon_set", "Orignal_Enroll_set", "Annon_Enroll_set", "trials"]
    required_order2 = ["Original_set", "Anon_set", "Original_Enroll_set", "Annon_Enroll_set", "trials"]
    missing_paths = []
    try:
        xls = pd.ExcelFile(xlsx_file)
        sheet_names = xls.sheet_names

        # ✅ Validate sheet order
        if sheet_names != required_order1 and sheet_names != required_order2:
            print("❌ Sheet names are not in the required order.")
            print(f"  Found:    {sheet_names}")
            print(f"  Expected: {required_order1}")
            print(f"  Expected: {required_order2}")
            sys.exit(1)
        else:
            print("✅ Sheet names are in the correct order.")

        # ✅ Validate wavpath existence
        for sheet_name in sheet_names:
            if sheet_name.lower() == 'trials':
                continue
            df = xls.parse(sheet_name)
            if 'wavpath' not in df.columns:
                print(f"Warning: 'wavpath' column not found in sheet '{sheet_name}'. Skipping...")
                continue
            for idx, path in enumerate(df['wavpath']):
                if not isinstance(path, str) or not os.path.exists(path):
                    missing_paths.append({
                        'sheet': sheet_name,
                        'row': idx + 2,  # +2 = header row + 1-based index
                        'path': path
                    })

        if missing_paths:
            print(f"❌ {len(missing_paths)} missing wav paths found:")
            for entry in missing_paths:
                print(f"  - Sheet: '{entry['sheet']}', Row: {entry['row']}, Path: {entry['path']}")
            sys.exit(1)
        else:
            print("✅ All wav paths exist (excluding 'trials' sheet).")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

def chunked(iterable, size):
    it = iter(iterable)
    while True:
        batch = list(islice(it, size))
        if not batch:
            break
        yield batch

def run_vp_eval(args):
    temp_name = args.out_path.replace('.xlsx', '')
    temp_dir = f'./tmp/tmp_{temp_name}'
    os.makedirs(temp_dir, exist_ok=True)

    df_path = args.protocol_path
    config_path = args.config_path
    language = args.language

    validate_wav_paths(df_path)

    asr_models = [
        "openai/whisper-large-v3"
    ]

    asv_models = [
        "asv_ecapa512"
    ]

    expected_outputs = []
    out_paths = {}
    futures = {}

    with concurrent.futures.ThreadPoolExecutor() as executor:
        # ASR in batches
        for batch in chunked(asr_models, 1):
            for model in batch:
                model_safe = model.replace("/", "_")
                out_path = os.path.join(temp_dir, f"asr_out_{model_safe}.csv")
                expected_outputs.append(out_path)
                if os.path.exists(out_path):
                    print(f"[SKIP] ASR model {model} already processed.")
                    out_paths[f"asr_{model_safe}"] = out_path
                    continue
                cmd = (
                    f"python src/asr.py --out_path {out_path} "
                    f"--config_path {config_path} --language '{language}' "
                    f"--protocol_path {df_path} --model {model}"
                )
                print(f"[INFO] Running ASR with model {model}")
                fut = executor.submit(subprocess.run, cmd, shell=True)
                futures[fut] = (cmd, out_path, f"asr_{model_safe}")

        # ASV
        for model in asv_models:
            model_safe = model.replace("/", "_")
            out_path = os.path.join(temp_dir, f"asv_out_{model_safe}.csv")
            expected_outputs.append(out_path)
            if os.path.exists(out_path):
                print(f"[SKIP] ASV model {model} already processed.")
                out_paths[f"asv_{model_safe}"] = out_path
                continue
            cmd = (
                f"python src/asv.py --out_path {out_path} "
                f"--config_path {config_path} --language '{language}' "
                f"--protocol_path {df_path} --model {model}"
            )
            print(f"[INFO] Running ASV with model {model}")
            fut = executor.submit(subprocess.run, cmd, shell=True)
            futures[fut] = (cmd, out_path, f"asv_{model_safe}")

        ### PROFILE START
        # Run Profile (Accent/Emotion cosine similarity)
        profile_out_path = os.path.join(temp_dir, "profile_out.csv")
        expected_outputs.append(profile_out_path)
        if os.path.exists(profile_out_path):
            print("[SKIP] Profile already processed.")
            out_paths["profile"] = profile_out_path
        else:
            cmd = (
                f"python src/run_profile.py --out_path {profile_out_path} "
                f"--config_path {config_path} --language '{language}' "
                f"--protocol_path {df_path}"
            )
            print(f"[INFO] Running PROFILE (accent/emotion/quality/fluency similarity)")
            fut = executor.submit(subprocess.run, cmd, shell=True)
            futures[fut] = (cmd, profile_out_path, "profile")
        ### PROFILE END

        # Generic and MOS
        for key in ["generic", "mos"]:
            out_path = os.path.join(temp_dir, f"{key}_out.csv")
            expected_outputs.append(out_path)
            if os.path.exists(out_path):
                print(f"[SKIP] {key.upper()} already processed.")
                out_paths[key] = out_path
                continue
            cmd = (
                f"python src/{key}.py --out_path {out_path} "
                f"--config_path {config_path} --language '{language}' "
                f"--protocol_path {df_path}"
            )
            print(f"[INFO] Running {key.upper()}")
            fut = executor.submit(subprocess.run, cmd, shell=True)
            futures[fut] = (cmd, out_path, key)

        # Collect results
        for fut, (cmd, out_path, key) in futures.items():
            try:
                result = fut.result()
                if result.returncode == 0:
                    print(f"[OK] Completed: {cmd}")
                    out_paths[key] = out_path
                else:
                    print(f"[FAIL] Command failed (exit {result.returncode}): {cmd}")
            except Exception as e:
                print(f"[EXCEPTION] Error running command {cmd}: {e}")

    # Verify all expected outputs exist
    missing_outputs = [p for p in expected_outputs if not os.path.exists(p)]
    if missing_outputs:
        print(f"⚠️ Missing outputs: {len(missing_outputs)} -> {missing_outputs}")
        return None  # signal failure

    # Compile results
    df_list = []
    for key, path in out_paths.items():
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                df_list.append(df)
            except Exception as e:
                print(f"[WARN] Could not load CSV {path}: {e}")
        else:
            print(f"[WARN] Expected output missing: {path}")

    if not df_list:
        print("❌ No valid results to compile. Exiting.")
        return None

    out_df = pd.concat(df_list, axis=1)
    return out_df


def main(args):
    with open(args.config_path) as f:
        _ = json.load(f)

    max_attempts = 5
    out_df = None
    for attempt in range(1, max_attempts+1):
        print(f"\n===== Attempt {attempt}/{max_attempts} =====")
        out_df = run_vp_eval(args)
        if out_df is not None:
            break
        else:
            print(f"⚠️ Attempt {attempt} failed. Retrying...")

    if out_df is None:
        print("❌ All attempts failed. Exiting.")
        sys.exit(1)

    # Save results after successful attempt
    out_csv_path = args.out_path.replace('.xlsx', '.csv')
    out_df.to_csv(out_csv_path, index=False)

    workbook = pd.ExcelWriter(args.out_path, engine='openpyxl')
    average_sheet = out_df.filter(regex='(avg|eer)', axis=1)
    average_sheet.to_excel(workbook, sheet_name='Average', index=False)
    rest_sheet = out_df.drop(average_sheet.columns, axis=1)
    rest_sheet.to_excel(workbook, sheet_name='Rest', index=False)
    workbook.close()

    print(f"✅ Results saved to {args.out_path} and {out_csv_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_path', default='test_out1.xlsx')
    parser.add_argument('--config_path', default='./configs/privacy_config.json')
    parser.add_argument('--language', default='English')
    parser.add_argument('--protocol_path', default='mcadams_globe_protocols.xlsx')

    args = parser.parse_args()
    main(args)
