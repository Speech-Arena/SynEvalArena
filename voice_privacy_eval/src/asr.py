# -*- coding: utf-8 -*-
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from datasets import Audio, Dataset
import pandas as pd
from tqdm import tqdm
import re
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from jiwer import wer, cer


class WhisperASR:
    def __init__(self, model_name="openai/whisper-tiny", lang="english", device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = WhisperProcessor.from_pretrained(model_name)
        self.model = WhisperForConditionalGeneration.from_pretrained(
            model_name, device_map=None, torch_dtype=torch.float32
        )
        self.model.to(self.device)

        self.lang = lang.lower()
        if self.lang != "mls":  # pre-defined language
            self.forced_decoder_ids = self.processor.get_decoder_prompt_ids(language=self.lang, task="transcribe")
            self.model.config.forced_decoder_ids = self.forced_decoder_ids
        else:
            self.forced_decoder_ids = None  # dynamic detection mode

    def init_dataset(self, col_list):
        dataset = Dataset.from_dict({'audio': col_list}).cast_column("audio", Audio(sampling_rate=16000))
        return dataset

    @torch.no_grad()
    def transcribe(self, col_list):
        dataset = self.init_dataset(col_list)
        transcriptions = []

        for i in dataset:
            inputs = self.processor(
                i["audio"]["array"],
                sampling_rate=i['audio']["sampling_rate"],
                return_tensors="pt"
            ).input_features
            inputs = inputs.to(self.device)

            if self.lang == "mls":
                # Let Whisper auto-detect language
                predicted_ids = self.model.generate(inputs)
                transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

                # detect language token
                detected_lang_id = predicted_ids[0][0].item()
                detected_lang = self.processor.tokenizer.convert_ids_to_tokens(detected_lang_id)

                # re-run if english to force english decoding (better accuracy)
                if "en" in detected_lang.lower():
                    forced_ids = self.processor.get_decoder_prompt_ids(language="english", task="transcribe")
                    predicted_ids = self.model.generate(inputs, forced_decoder_ids=forced_ids)
                    transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
            else:
                predicted_ids = self.model.generate(inputs, forced_decoder_ids=self.forced_decoder_ids)
                transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

            transcriptions.append(self._normalize(transcription))

        return transcriptions

    def _normalize(self, wrd):
        wrd = re.sub(r'[^\w\s]', '', wrd)
        return wrd.upper().strip()


def run_asr_evaluation_parallel(protocol_path, language, model):
    """
    Run ASR transcriptions for multiple Whisper models in parallel.
    Returns a DataFrame with reference texts and transcriptions from each model.
    """

    df = [pd.read_excel(protocol_path, sheet_name=i) for i in range(5)]

    args_list = [
        (model, language, df, f"orig_set_{model}_text", f"anon_set_{model}_text")
    ]

    results = []
    for model_args in args_list:
        result = run_asr_process(model_args)
        results.append(result)

    outdict = {
        "orig_set_text": df[0]['text'],
        "anon_set_text": df[1]['text']
    }
    for res in results:
        outdict.update(res)

    df = pd.DataFrame(outdict)

    def _normalize(wrd):
        wrd = re.sub(r'[^\w\s]', '', wrd)
        return wrd.upper().strip()

    df['orig_set_text'] = df['orig_set_text'].apply(_normalize)
    df['anon_set_text'] = df['anon_set_text'].apply(_normalize)

    df[f"orig_set_{model}_wer"] = df.apply(lambda x: wer(x['orig_set_text'], x[f'orig_set_{model}_text']), axis=1)
    df[f"orig_set_{model}_cer"] = df.apply(lambda x: cer(x['orig_set_text'], x[f'orig_set_{model}_text']), axis=1)

    df[f"anon_set_{model}_wer"] = df.apply(lambda x: wer(x['anon_set_text'], x[f'anon_set_{model}_text']), axis=1)
    df[f"anon_set_{model}_cer"] = df.apply(lambda x: cer(x['anon_set_text'], x[f'anon_set_{model}_text']), axis=1)

    outdict_ = {
        f"avg_orig_set_{model}_wer": round(df[f"orig_set_{model}_wer"].mean()*100, 2),
        f"avg_orig_set_{model}_cer": round(df[f"orig_set_{model}_cer"].mean()*100, 2),
        f"avg_anon_set_{model}_wer": round(df[f"anon_set_{model}_wer"].mean()*100, 2),
        f"avg_anon_set_{model}_cer": round(df[f"anon_set_{model}_cer"].mean()*100, 2)
    }

    df_ = pd.DataFrame(outdict_, index=[0])
    df = pd.concat([df, df_], axis=1)

    return df


def run_asr_process(model_args):
    model_name, language, df, col_name_orig, col_name_anon = model_args
    print(f"[INFO] Running ASR with model: {model_name} | Language: {language}")
    asr_infer = WhisperASR(model_name=model_name, lang=language)
    return {
        col_name_orig: asr_infer.transcribe(df[0]["wavpath"]),
        col_name_anon: asr_infer.transcribe(df[1]["wavpath"])
    }


if __name__ == '__main__':
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument('--out_path', default='test_out_asr.csv', required=False)
    parser.add_argument('--config_path', default='./configs/privacy_config_simple.json', required=False)
    parser.add_argument('--language', default='English', required=False, help="Set language or 'MLS' for auto-detect")
    parser.add_argument('--protocol_path', default='./protocols/librispeech/KNNVC-COS_librispeech_f.xlsx', required=False)
    parser.add_argument('--model', default='openai/whisper-large-v3', required=False)

    args = parser.parse_args()

    with open(args.config_path) as f:
        h = json.load(f)

    outdf = run_asr_evaluation_parallel(args.protocol_path, args.language, args.model)
    outdf.to_csv(args.out_path, index=False, float_format='%.3f')
