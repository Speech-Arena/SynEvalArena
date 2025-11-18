
import torch
from datasets import Audio, Dataset
import seaborn as sns 
import matplotlib.pyplot as plt 
from matplotlib.ticker import MaxNLocator

import pandas as pd
from jiwer import process_words
from collections import Counter
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns

from transformers import (
    WhisperForConditionalGeneration, 
    WhisperProcessor,
    )
from tqdm import tqdm
import jiwer
from jiwer import wer, cer

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from g2p_en import G2p
import re

# Cache for loaded models/tokenizers
_g2p_models = {}

def load_model(lang_code):
    model_map = {
        'fr': 'kabalab/g2p-fr',
        'de': 'kabalab/g2p-de',
        'nl': 'kabalab/g2p-nl',
        'it': 'kabalab/g2p-it'
    }
    model_name = model_map[lang_code]
    if lang_code not in _g2p_models:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        _g2p_models[lang_code] = (tokenizer, model)
    return _g2p_models[lang_code]

def g2p_text(text: str, lang: str) -> str:
    lang = lang.lower()
    if not isinstance(text, str) or text.strip() == "":
        return ""

    if lang == 'en':
        g2p = G2p()
        phonemes = g2p(text)
        return " ".join([p for p in phonemes if p not in [' ', '▁']])
    
    elif lang in ['fr', 'de', 'it', 'nl']:
        tokenizer, model = load_model(lang)
        inputs = tokenizer(text, return_tensors="pt")
        outputs = model.generate(**inputs)
        phoneme_str = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return phoneme_str
    
    else:
        return "[unsupported language]"




def compute_wer(row, pred_colname):
    """
    Utility function for pandas df.apply to cpmpute row wise WER
    """
    return wer(row.ground_truth_transcript, row[pred_colname])*100

def compute_cer(row, pred_colname):
    """
    Utility function for pandas df.apply to cpmpute row wise CER
    """
    return cer(row.ground_truth_transcript, row[pred_colname])*100

def compute_per(row, pred_colname):
    """
    Utility function for pandas df.apply to cpmpute row wise PER
    """
    return cer(row.g2p_ground_truth, row[pred_colname])*100

def inference_whisper(df, pred_colname, prefix, size='large-v3', lang='english'):
    """
    ASR Inference using OpenAI Whisper Large-v3

    Inputs :
        df - Pandas data frame containing ground truth transcript along with ref and synth audio.
        pred_colname - Whether to do inference for ref or synth audio. (ref_audio_path or syn_audio_path)
        prefix - prefix used to indicate predictions are for ref or syn audios (ref_ or syn_)
        size - Model size of whisper to usee. Default large-v3.
        lang - Inference lang. Options - english, french, german, italian

    Returns : 
        df - Same data frame with a new column - whisper_transcription prefixed with prefix.s

    """

    def _normalize(wrd):
        wrd = re.sub(r'[^\w\s]', '', wrd)
        return wrd.upper()

    if torch.cuda.is_available():
        device= torch.device('cuda')
    else:
        device=torch.device('cpu')


    dataset = Dataset.from_dict({'audio' : df[pred_colname]}).cast_column("audio", Audio(sampling_rate=16000))

    processor = WhisperProcessor.from_pretrained(f"openai/whisper-{size}")
    model = WhisperForConditionalGeneration.from_pretrained(f"openai/whisper-{size}")

    forced_decoder_ids = processor.get_decoder_prompt_ids(language=lang, task="transcribe")

    model.to(device)

    transcriptions = []
    for i in tqdm(dataset):
        with torch.no_grad():
            inputs = processor(i["audio"]["array"], sampling_rate=i['audio']["sampling_rate"], return_tensors="pt").input_features
            inputs = inputs.to(device)
            predicted_ids = model.generate(inputs, forced_decoder_ids=forced_decoder_ids)
            transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
            transcriptions.append(_normalize(transcription[0]))
    df[f'{prefix}whisper_transcription'] = transcriptions

    return df

def compute_asr_metrics(df, prefix, pred_colname, lang):
    """
    Computes ASR metrics - WER, PER, CER
    Also does g2p using espeak

    Inputs :
        df - Pandas data frame containing ground truth transcript along with ref and synth audio.
        pred_colname - Whether to do inference for ref or synth audio. (ref_audio_path or syn_audio_path)
        prefix - prefix used to indicate predictions are for ref or syn audios (ref_ or syn_)
        size - Model size of whisper to usee. Default large-v3.
        lang - Inference lang. Options - english, french, german, italian

    Returns : 
        df - Same data frame with new columns
            wer, cer, per - prefixed with prefix
            g2p_ground_truth - g2p performed on ground truth transcript
            g2p_predictions - prefixed with prefix for ref / syn.

    """

    #separator = Separator(phone=' ', word=None)
    lang_map = {'english':'en', 'french':'fr', 'italian':'it', 'german':'de'}
    language = lang_map[lang]

    df[f'{prefix}g2p_predictions'] = df[pred_colname].apply(lambda x: g2p_text(x, language))

    df['g2p_ground_truth'] = df.ground_truth_transcript.apply(lambda x: g2p_text(x, language))
    
    df[f'{prefix}wer'] = df.apply(compute_wer, axis=1, args=(pred_colname, ))
    df[f'{prefix}cer'] = df.apply(compute_cer, axis=1, args=(pred_colname, ))
    df[f'{prefix}per'] = df.apply(compute_per, axis=1, args=(f'{prefix}g2p_predictions', ))

    return df

def asr_inference(df, lang):
    """
    Wrapper function that does asr inference and computes metrics using inference_whisper and compute_asr_metrics
    """
    df = inference_whisper(df, pred_colname='syn_audio_path', prefix='syn_', lang=lang)
    #df = inference_whisper(df, pred_colname='ref_audio_path', prefix='ref_', lang=lang)
    #df = compute_asr_metrics(df, prefix='ref_', pred_colname='ref_whisper_transcription', lang=lang)
    df = compute_asr_metrics(df, prefix='syn_', pred_colname='syn_whisper_transcription', lang=lang)

    return df
    
    
def extract_diphones(text):
    text = text.replace(" ", "").replace("\n", "")
    return ' '.join(f"{a}-{b}" for a, b in zip(text, text[1:])) if len(text) > 1 else ""

def count_phoneme_errors(df, pred_colname, topk=50, diphones=False):
    gt_col = 'g2p_ground_truth'

    if diphones:
        gt_diphones = df[gt_col].map(extract_diphones)
        pred_diphones = df[pred_colname].map(extract_diphones)
    else:
        gt_diphones = df[gt_col]
        pred_diphones = df[pred_colname]

    error_counts = Counter()

    for ref, hyp in zip(gt_diphones, pred_diphones):
        ref_tokens, hyp_tokens = ref.split(), hyp.split()
        for a in process_words(ref, hyp).alignments[0]:
            if a.type == 'equal':
                continue
            indices = range(a.ref_start_idx, a.ref_end_idx) if a.type != 'insert' else range(a.hyp_start_idx, a.hyp_end_idx)
            tokens = ref_tokens if a.type != 'insert' else hyp_tokens
            error_counts.update(tokens[i] for i in indices)

    return dict(error_counts.most_common(topk))


def plot_phonemes_util(map_1, map_2, map_3, map_4):
    fig, axs = plt.subplots(2, 2, figsize=(40, 20))
    fig.suptitle('Phoneme and Diphone Error Frequency Analysis', fontsize=24)
    palette = sns.color_palette("Set2", 10)

    titles = ['Phonemes (reference)', 'Phonemes (synthesized)', 'Diphones (reference)', 'Diphones (synthesized)']
    for ax, data, title in zip(axs.flatten(), [map_1, map_2, map_3, map_4], titles):
        ax.bar(data.keys(), data.values(), color=palette)
        ax.set_xlabel(title)
        ax.set_ylabel('Frequency of errors (I + S + D)')
        ax.tick_params(labelrotation=90)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    plt.tight_layout()


def plot_phonemes(df, outfile_path, topk=50):
    map_phn_ref = count_phoneme_errors(df, 'ref_g2p_predictions', topk, diphones=False)
    map_phn_syn = count_phoneme_errors(df, 'syn_g2p_predictions', topk, diphones=False)
    map_diph_ref = count_phoneme_errors(df, 'ref_g2p_predictions', topk, diphones=True)
    map_diph_syn = count_phoneme_errors(df, 'syn_g2p_predictions', topk, diphones=True)
    plot_phonemes_util(map_phn_ref, map_phn_syn, map_diph_ref, map_diph_syn)
    plt.savefig(outfile_path)



