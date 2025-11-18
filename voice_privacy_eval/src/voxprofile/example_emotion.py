import torch
import torchaudio
import torch.nn.functional as F
from src.model.emotion.whisper_emotion import WhisperWrapper

emotion_list = [
    'Anger', 
    'Contempt', 
    'Disgust', 
    'Fear', 
    'Happiness', 
    'Neutral', 
    'Sadness', 
    'Surprise', 
    'Other'
]
    
# Find device
device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

# Load model from Huggingface
model = WhisperWrapper.from_pretrained("tiantiaf/whisper-large-v3-msp-podcast-emotion").to(device)
model.eval()

# Load data, here just zeros as the example
# Our training data filters output audio shorter than 3 seconds (unreliable predictions) and longer than 15 seconds (computation limitation)
# So you need to prepare your audio to a maximum of 15 seconds, 16kHz and mono channel
max_audio_length = 15 * 16000
data, _ = torchaudio.load("YOUR_DATA")
data = data.float().to(device)[:, :max_audio_length]
logits, embedding, _, _, _, _ = model(data, return_feature=True)
    
# Probability
emotion_prob = F.softmax(logits, dim=1)
print(emotion_list[torch.argmax(emotion_prob).detach().cpu().item()])