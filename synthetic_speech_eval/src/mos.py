from src.wvmos import get_wvmos
import torch
import librosa
import soundfile as sf
import utmosv2

def compute_wvmos(df):
    """
    Computes speech MOS using wvmos : https://github.com/AndreevP/wvmos/blob/main/wvmos/wv_mos.py

    Inputs :
        df - Pandas data frame containing reference and synthesized audio paths

    Returns : 
        df - Same data frame with two new columns - wvmos_syn & wvmos_ref for predicted MOS Scores.
    """

    if torch.cuda.is_available():
        use_cuda = True
    else:
        use_cuda = False

    model = get_wvmos(cuda=use_cuda)   
    wvmos_syn = []
    wvmos_ref = []

    for i in range(df.shape[0]):
        syn = model.calculate_one(df.syn_audio_path[i])
        ref = model.calculate_one(df.ref_audio_path[i]) 
        wvmos_syn.append(syn)
        wvmos_ref.append(ref)

    df['wvmos_syn'] = wvmos_syn
    df['wvmos_ref'] = wvmos_ref

    return df

def compute_speech_mos(df):
    """
    Computes speech MOS using SpeechMOS : https://github.com/tarepan/SpeechMOS

    Inputs :
        df - Pandas data frame containing reference and synthesized audio paths

    Returns : 
        df - Same data frame with two new columns - speech_mos_syn & speech_mos_ref for predicted MOS Scores.
    """

    predictor = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True)
    speech_mos_syn = []
    speech_mos_ref = []
    for i in range(df.shape[0]):
        wave_syn, sr = librosa.load(df.syn_audio_path[i], sr=None, mono=True)
        syn = predictor(torch.from_numpy(wave_syn).unsqueeze(0), sr).item()
        wave_ref, sr = librosa.load(df.ref_audio_path[i], sr=None, mono=True)
        ref = predictor(torch.from_numpy(wave_ref).unsqueeze(0), sr).item()
        speech_mos_syn.append(syn)
        speech_mos_ref.append(ref)
    
    df['speech_mos_syn'] = speech_mos_syn
    df['speech_mos_ref'] = speech_mos_ref

    return df


def compute_utmos2_mos(df):
    """
    
    """

    predictor = utmosv2.create_model(pretrained=True)
    mos_syn = []
    mos_ref = []
    for i in range(df.shape[0]):
        syn = predictor.predict(input_path=df.syn_audio_path[i])
        ref = predictor.predict(input_path=df.ref_audio_path[i])
        mos_syn.append(syn)
        mos_ref.append(ref)
    
    df['utmos2_mos_syn'] = mos_syn
    df['utmos2_mos_ref'] = mos_ref

    return df




