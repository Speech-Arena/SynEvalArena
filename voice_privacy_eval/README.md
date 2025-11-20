![Syn Eval Tool](../assets/syn-eval-tool.jpg)

## Proposed metrics


For unbiased evaluation of privacy and utility in voice anonymization, we propose the following metrics:

- **AUDI** — ASR Utility Distortion Index (lower is better)  
- **Fusion EER** — Equal Error Rate from fusion of multiple ASV systems (higher is better)

---

#### 1. ASR Utility Distortion Index (AUDI)

For a given utterance <img src="https://latex.codecogs.com/svg.image?\color{white}u">, anonymization system <img src="https://latex.codecogs.com/svg.image?\color{white}s">, and a set of ASR systems <img src="https://latex.codecogs.com/svg.image?\color{white}\mathcal{A}">, we define:  

- <img src="https://latex.codecogs.com/svg.image?\color{white}\mathrm{WER}^{a}_{\text{orig}}(u)=\text{WER%20between%20reference%20original%20transcription%20and%20ASR%20output%20from%20system%20}a">  
- <img src="https://latex.codecogs.com/svg.image?\color{white}\mathrm{WER}^{a}_{\text{anon}}(u)=\text{WER%20between%20anonymized%20transcription%20and%20ASR%20output%20from%20system%20}a">

Then, for utterance <img src="https://latex.codecogs.com/svg.image?\color{white}u">:

- <img src="https://latex.codecogs.com/svg.image?\color{white}\Delta^{a}(u)=\big|\mathrm{WER}^{a}_{\text{orig}}(u)-\mathrm{WER}^{a}_{\text{anon}}(u)\big|">

The DSII score for utterance <img src="https://latex.codecogs.com/svg.image?\color{white}u">:

- <img src="https://latex.codecogs.com/svg.image?\color{white}\mathrm{DSII}(u)=\sum_{a\in\mathcal{A}}\Delta^{a}(u)">

The mean DSII over dataset <img src="https://latex.codecogs.com/svg.image?\color{white}\mathcal{U}">:

- <img src="https://latex.codecogs.com/svg.image?\color{white}\mathrm{DSII}=\frac{1}{|\mathcal{U}|}\sum_{u\in\mathcal{U}}\mathrm{DSII}(u)">

---

#### 2. Fusion Equal Error Rate (EER)

Given:  
- <img src="https://latex.codecogs.com/svg.image?\color{white}N"> systems producing scores <img src="https://latex.codecogs.com/svg.image?\color{white}s_i(n)"> for trial <img src="https://latex.codecogs.com/svg.image?\color{white}n">  
- Labels <img src="https://latex.codecogs.com/svg.image?\color{white}y(n)"> where:  
  - <img src="https://latex.codecogs.com/svg.image?\color{white}y(n)=1"> → impostor trial  
  - <img src="https://latex.codecogs.com/svg.image?\color{white}y(n)=0"> → genuine trial  

#### Fusion score for trial <img src="https://latex.codecogs.com/svg.image?\color{white}n">

- <img src="https://latex.codecogs.com/svg.image?\color{white}\mathrm{s}_\text{fusion}(n)=\frac{1}{\mathcal{N}}\sum_{i=1..N}\mathrm{s}_{\text{i}}(n)">

#### ASR Systems
- Whisper Tiny  
- Whisper Base  
- Whisper Small  
- Whisper Medium  
- Whisper Large  
- Whisper Large V3  

#### ASV Systems
- ECAPA-TDNN 1024  
- ECAPA-TDNN 512  
- ResNet 34  
- ResNet 152  

---

## Usage     

###### 1. Create your env

```bash
conda create --name vp python=3.10
conda activate vp
pip install -r requirements.txt
cd ./src/voxprofile/
pip install -e .
```

###### 2. Setting up

- Step 1. edit config file inside `/sq/configs/privacy_configs.json`
modify directory paths and protocol xlsx filepaths

- Step 2. run following command

```bash
python main_vp.py --protocol_path /idiap/temp/akulkarni/vp_work/eval_tool/sq/protocols/mls/knnvcr_MLS_vp_protocol.xlsx --out_path knnvcr_MLS_vp_protocol_results.xlsx
```

```bash
python main_single.py --protocol_path /idiap/temp/akulkarni/vp_work/eval_tool/sq/protocols/mls/knnvcr_MLS_vp_protocol.xlsx --out_path knnvcr_MLS_vp_protocol_results.xlsx
```
###### Pre-requisites:

Download following datasets:
- LibriSpeech dev and test
https://www.openslr.org/resources/12/dev-clean.tar.gz
https://www.openslr.org/resources/12/test-clean.tar.gz

- Samrómur Children 21.09 
https://www.openslr.org/resources/117/samromur_children_21.09.zip

- SpeechOcean 762
https://www.openslr.org/resources/101/speechocean762.tar.gz

- PF STAR















