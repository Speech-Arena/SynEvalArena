![Syn Eval Tool](../assets/syn-eval-tool.jpg)

## Proposed metrics


For unbiased evaluation of privacy and utility in voice anonymization, we propose the following metrics:

- **AUDI** — ASR Utility Distortion Index (lower is better)  
- **Fusion EER** — Equal Error Rate from fusion of multiple ASV systems (higher is better)

---

#### 1. ASR Utility Distortion Index (AUDI)

For a given utterance <img src="https://latex.codecogs.com/svg.latex?u">, anonymization system <img src="https://latex.codecogs.com/svg.latex?s">, and a set of ASR systems <img src="https://latex.codecogs.com/svg.latex?%5Cmathcal%7BA%7D">, we define:  

- <img src="https://latex.codecogs.com/svg.latex?%5Cmathrm%7BWER%7D%5E%7Ba%7D_%7B%5Ctext%7Borig%7D%7D(u)%20=%20%5Ctext%7BWER%20between%20reference%20original%20transcription%20and%20ASR%20output%20from%20system%20%7D%20a">  
- <img src="https://latex.codecogs.com/svg.latex?%5Cmathrm%7BWER%7D%5E%7Ba%7D_%7B%5Ctext%7Banon%7D%7D(u)%20=%20%5Ctext%7BWER%20between%20anonymized%20transcription%20and%20ASR%20output%20from%20system%20%7D%20a">

Then, for utterance <img src="https://latex.codecogs.com/svg.latex?u">:

- <img src="https://latex.codecogs.com/svg.latex?%5CDelta%5E%7Ba%7D(u)%20=%20%5Cbig%7C%20%5Cmathrm%7BWER%7D%5E%7Ba%7D_%7B%5Ctext%7Borig%7D%7D(u)%20-%20%5Cmathrm%7BWER%7D%5E%7Ba%7D_%7B%5Ctext%7Banon%7D%7D(u)%20%5Cbig%7C">

The DSII score for utterance <img src="https://latex.codecogs.com/svg.latex?u">:

- <img src="https://latex.codecogs.com/svg.latex?%5Cmathrm%7BDSII%7D(u)%20=%20%5Csum_%7Ba%20%5Cin%20%5Cmathcal%7BA%7D%7D%20%5CDelta%5E%7Ba%7D(u)">

The mean DSII over dataset <img src="https://latex.codecogs.com/svg.latex?%5Cmathcal%7BU%7D">:

- <img src="https://latex.codecogs.com/svg.latex?%5Cmathrm%7BDSII%7D%20=%20%5Cfrac%7B1%7D%7B%7C%5Cmathcal%7BU%7D%7C%7D%20%5Csum_%7Bu%20%5Cin%20%5Cmathcal%7BU%7D%7D%20%5Cmathrm%7BDSII%7D(u)">

---

#### 2. Fusion Equal Error Rate (EER)

Given:  
- <img src="https://latex.codecogs.com/svg.latex?N"> systems producing scores <img src="https://latex.codecogs.com/svg.latex?s_i(n)"> for trial <img src="https://latex.codecogs.com/svg.latex?n">  
- Labels <img src="https://latex.codecogs.com/svg.latex?y(n)"> where:  
  - <img src="https://latex.codecogs.com/svg.latex?y(n)=1"> → impostor trial  
  - <img src="https://latex.codecogs.com/svg.latex?y(n)=0"> → genuine trial  

#### Fusion score for trial <img src="https://latex.codecogs.com/svg.latex?n">:

<img src="https://latex.codecogs.com/svg.latex?s_%7B%5Ctext%7Bfusion%7D%7D(n)%20=%20%5Cfrac%7B1%7D%7BN%7D%20%5Csum_%7Bi=1%7D%5EN%20s_i(n)">

---

We use the following ASR and ASV systems for computing the metrics:  

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















