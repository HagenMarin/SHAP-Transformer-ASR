import torch
from transformers import AutoModel
import numpy as np
import matplotlib.pyplot as plt
import torchaudio
from torchaudio.models import Conformer
from datasets import load_dataset
import shap
import logging
import sys
import os
import soundfile as sf
import nemo.collections.asr as nemo_asr

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")


asr_model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained("nvidia/stt_en_conformer_ctc_large")
print(asr_model)

raw_audio, sample_rate = torchaudio.load('2086-149220-0033.wav')

print(raw_audio.flatten().shape)

processed_signal, processed_signal_length = asr_model.preprocessor(
    input_signal=raw_audio.to(device=device),
    length=torch.tensor(raw_audio.shape[1]).unsqueeze(0).to(device=device),
)


output = asr_model.forward(processed_signal=processed_signal, processed_signal_length=processed_signal_length)
print(output)

hypotheses = asr_model.decoding.ctc_decoder_predictions_tensor(
            output[0],
            decoder_lengths=output[1],
            return_hypotheses=True,
        )

print(hypotheses)
