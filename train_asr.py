import torch
import torchaudio
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import numpy as np
from pathlib import Path
import random
import librosa
import soundfile as sf
from typing import List, Tuple, Dict
import logging
import os
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NoiseDataset:
    def __init__(self, noise_type: str, noise_dir: str):
        """
        Initialize noise dataset for different noise types
        
        Args:
            noise_type: Type of noise ('chime', 'ssn', or 'network')
            noise_dir: Directory containing noise files
        """
        self.noise_type = noise_type
        self.noise_dir = Path(noise_dir)
        self.noise_files = list(self.noise_dir.glob('**/*.wav'))
        
    def get_noise(self, duration: float, sample_rate: int) -> torch.Tensor:
        """
        Get noise sample of specified duration
        
        Args:
            duration: Duration in seconds
            sample_rate: Audio sample rate
            
        Returns:
            torch.Tensor: Noise waveform
        """
        if self.noise_type == 'chime':
            return self._get_chime_noise(duration, sample_rate)
        elif self.noise_type == 'ssn':
            return self._get_ssn_noise(duration, sample_rate)
        elif self.noise_type == 'network':
            return self._get_network_noise(duration, sample_rate)
        else:
            raise ValueError(f"Unknown noise type: {self.noise_type}")
    
    def _get_chime_noise(self, duration: float, sample_rate: int) -> torch.Tensor:
        """Get CHiME noise sample"""
        noise_file = random.choice(self.noise_files)
        waveform, sr = torchaudio.load(noise_file)
        if sr != sample_rate:
            waveform = torchaudio.transforms.Resample(sr, sample_rate)(waveform)
        
        # Ensure noise is long enough
        if waveform.shape[1] < duration * sample_rate:
            waveform = waveform.repeat(1, int(np.ceil(duration * sample_rate / waveform.shape[1])))
        
        # Trim to exact duration
        waveform = waveform[:, :int(duration * sample_rate)]
        return waveform
    
    def _get_ssn_noise(self, duration: float, sample_rate: int) -> torch.Tensor:
        """Generate Speech Shaped Noise"""
        # Generate white noise
        white_noise = torch.randn(1, int(duration * sample_rate))
        
        # Get 6 random clean speech files for filter
        clean_speech_dir = Path("path/to/clean/speech")  # Update with actual path
        clean_files = random.sample(list(clean_speech_dir.glob('*.wav')), 6)
        
        # Compute average magnitude spectrum
        avg_mag = torch.zeros(1, 1 + sample_rate // 2)
        for file in clean_files:
            audio, _ = torchaudio.load(file)
            spec = torch.stft(audio, n_fft=2048, return_complex=True)
            avg_mag += torch.abs(spec).mean(dim=-1)
        avg_mag /= len(clean_files)
        
        # Apply filter
        filtered_noise = torch.istft(
            torch.stft(white_noise, n_fft=2048, return_complex=True) * avg_mag.unsqueeze(-1),
            n_fft=2048
        )
        return filtered_noise
    
    def _get_network_noise(self, duration: float, sample_rate: int) -> torch.Tensor:
        """Get Network Sound Effects noise"""
        noise_file = random.choice(self.noise_files)
        waveform, sr = torchaudio.load(noise_file)
        if sr != sample_rate:
            waveform = torchaudio.transforms.Resample(sr, sample_rate)(waveform)
        
        # Ensure noise is long enough
        if waveform.shape[1] < duration * sample_rate:
            waveform = waveform.repeat(1, int(np.ceil(duration * sample_rate / waveform.shape[1])))
        
        # Trim to exact duration
        waveform = waveform[:, :int(duration * sample_rate)]
        return waveform

class NoisySpeechDataset(Dataset):
    def __init__(
        self,
        clean_speech_dir: str,
        noise_datasets: Dict[str, NoiseDataset],
        processor: Wav2Vec2Processor,
        snr_levels: List[float] = [-5, 0, 5, 10, 15],
        sample_rate: int = 16000
    ):
        """
        Dataset for noisy speech with multiple noise types
        
        Args:
            clean_speech_dir: Directory containing clean speech files
            noise_datasets: Dictionary of NoiseDataset objects for different noise types
            processor: Wav2Vec2 processor for tokenization
            snr_levels: List of SNR levels to use
            sample_rate: Audio sample rate
        """
        self.clean_speech_dir = Path(clean_speech_dir)
        self.clean_files = list(self.clean_speech_dir.glob('**/*.wav'))
        self.noise_datasets = noise_datasets
        self.processor = processor
        self.snr_levels = snr_levels
        self.sample_rate = sample_rate
        
    def __len__(self) -> int:
        return len(self.clean_files)
    
    def __getitem__(self, idx: int) -> Dict:
        # Load clean speech
        clean_file = self.clean_files[idx]
        clean_waveform, sr = torchaudio.load(clean_file)
        if sr != self.sample_rate:
            clean_waveform = torchaudio.transforms.Resample(sr, self.sample_rate)(clean_waveform)
        
        # Get random noise type and SNR
        noise_type = random.choice(list(self.noise_datasets.keys()))
        snr = random.choice(self.snr_levels)
        
        # Get noise sample
        noise = self.noise_datasets[noise_type].get_noise(
            clean_waveform.shape[1] / self.sample_rate,
            self.sample_rate
        )
        
        # Mix speech and noise at specified SNR
        noisy_waveform = self._mix_at_snr(clean_waveform, noise, snr)
        
        # Process for model input
        inputs = self.processor(
            noisy_waveform.squeeze().numpy(),
            sampling_rate=self.sample_rate,
            return_tensors="pt"
        )
        
        return {
            "input_values": inputs.input_values.squeeze(),
            "attention_mask": inputs.attention_mask.squeeze(),
            "noise_type": noise_type,
            "snr": snr
        }
    
    def _mix_at_snr(self, speech: torch.Tensor, noise: torch.Tensor, snr: float) -> torch.Tensor:
        """Mix speech and noise at specified SNR level"""
        speech_power = torch.mean(speech ** 2)
        noise_power = torch.mean(noise ** 2)
        
        # Calculate scaling factor for noise
        scale = torch.sqrt(speech_power / (noise_power * 10 ** (snr / 10)))
        
        # Mix signals
        noisy_speech = speech + scale * noise
        return noisy_speech

class EarlyStopping:
    """Early stopping handler"""
    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 0.0,
        mode: str = 'min'
    ):
        """
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change in monitored value to qualify as an improvement
            mode: 'min' for minimizing loss, 'max' for maximizing metrics
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_value = None
        self.should_stop = False
        
    def __call__(self, value: float) -> bool:
        if self.best_value is None:
            self.best_value = value
        elif self.mode == 'min':
            if value > self.best_value - self.min_delta:
                self.counter += 1
            else:
                self.best_value = value
                self.counter = 0
        else:  # mode == 'max'
            if value < self.best_value + self.min_delta:
                self.counter += 1
            else:
                self.best_value = value
                self.counter = 0
                
        if self.counter >= self.patience:
            self.should_stop = True
            
        return self.should_stop

class ModelCheckpoint:
    """Model checkpoint handler"""
    def __init__(
        self,
        save_dir: str,
        monitor: str = 'loss',
        mode: str = 'min',
        save_best_only: bool = True
    ):
        """
        Args:
            save_dir: Directory to save checkpoints
            monitor: Metric to monitor ('loss' or 'accuracy')
            mode: 'min' for minimizing loss, 'max' for maximizing metrics
            save_best_only: If True, only save the best model
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.best_value = float('inf') if mode == 'min' else float('-inf')
        
    def __call__(
        self,
        model: nn.Module,
        value: float,
        epoch: int,
        optimizer: torch.optim.Optimizer,
        loss: float
    ):
        if self.save_best_only:
            if (self.mode == 'min' and value < self.best_value) or \
               (self.mode == 'max' and value > self.best_value):
                self.best_value = value
                self._save_checkpoint(model, epoch, optimizer, loss, is_best=True)
        else:
            self._save_checkpoint(model, epoch, optimizer, loss, is_best=False)
            
    def _save_checkpoint(
        self,
        model: nn.Module,
        epoch: int,
        optimizer: torch.optim.Optimizer,
        loss: float,
        is_best: bool
    ):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'best_value': self.best_value
        }
        
        # Save regular checkpoint
        checkpoint_path = self.save_dir / f'checkpoint_epoch_{epoch}.pt'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model if applicable
        if is_best:
            best_model_path = self.save_dir / 'best_model.pt'
            torch.save(checkpoint, best_model_path)
            logger.info(f"New best model saved with value: {self.best_value:.4f}")

def train(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_epochs: int,
    early_stopping: EarlyStopping = None,
    checkpoint: ModelCheckpoint = None,
    val_loader: DataLoader = None
):
    """
    Training loop with early stopping and checkpointing
    
    Args:
        model: Model to train
        train_loader: Training data loader
        optimizer: Optimizer
        device: Device to train on
        num_epochs: Number of epochs to train
        early_stopping: Early stopping handler
        checkpoint: Model checkpoint handler
        val_loader: Validation data loader
    """
    model.train()
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        total_loss = 0
        for batch in train_loader:
            input_values = batch["input_values"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            optimizer.zero_grad()
            outputs = model(input_values, attention_mask=attention_mask)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_train_loss = total_loss / len(train_loader)
        logger.info(f"Epoch {epoch+1}/{num_epochs}, Average Training Loss: {avg_train_loss:.4f}")
        
        # Validation phase
        if val_loader is not None:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    input_values = batch["input_values"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    
                    outputs = model(input_values, attention_mask=attention_mask)
                    val_loss += outputs.loss.item()
            
            avg_val_loss = val_loss / len(val_loader)
            logger.info(f"Validation Loss: {avg_val_loss:.4f}")
            
            # Checkpoint
            if checkpoint is not None:
                checkpoint(model, avg_val_loss, epoch, optimizer, avg_train_loss)
            
            # Early stopping
            if early_stopping is not None:
                if early_stopping(avg_val_loss):
                    logger.info("Early stopping triggered")
                    break
        else:
            # If no validation set, use training loss for checkpointing
            if checkpoint is not None:
                checkpoint(model, avg_train_loss, epoch, optimizer, avg_train_loss)
            
            # Early stopping on training loss
            if early_stopping is not None:
                if early_stopping(avg_train_loss):
                    logger.info("Early stopping triggered")
                    break

def main():
    # Initialize processor and model
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
    
    # Initialize noise datasets
    noise_datasets = {
        "chime": NoiseDataset("chime", "path/to/chime/noise"),
        "ssn": NoiseDataset("ssn", "path/to/ssn/noise"),
        "network": NoiseDataset("network", "path/to/network/noise")
    }
    
    # Create datasets and dataloaders
    train_dataset = NoisySpeechDataset(
        clean_speech_dir="path/to/clean/speech",
        noise_datasets=noise_datasets,
        processor=processor
    )
    
    # Split dataset into train and validation
    train_size = int(0.9 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        num_workers=4
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=4
    )
    
    # Training setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    # Initialize early stopping and checkpointing
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_dir = Path("checkpoints") / timestamp
    early_stopping = EarlyStopping(patience=5, min_delta=0.001)
    checkpoint = ModelCheckpoint(
        save_dir=str(checkpoint_dir),
        monitor='loss',
        mode='min',
        save_best_only=True
    )
    
    # Train model
    train(
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        device=device,
        num_epochs=10,
        early_stopping=early_stopping,
        checkpoint=checkpoint,
        val_loader=val_loader
    )

if __name__ == "__main__":
    main() 