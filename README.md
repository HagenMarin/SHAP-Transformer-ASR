# SHAP-Transformer-ASR

Exploring SHAP for interpreting Transformer-based ASR models. This project provides explainability via Shapley values to improve transparency in speech recognition. It includes code, benchmarks, and analysis for debugging and building trust in high-stakes applications.

## Overview

SHAP-Transformer-ASR integrates SHAP (SHapley Additive exPlanations) with Transformer-based Automatic Speech Recognition (ASR) models, such as Wav2Vec2. The project focuses on providing insights into model behavior by analyzing feature importance, visualizing SHAP values, and evaluating their correlation with model performance metrics like Word Error Rate (WER).

Key features include:
- SHAP value computation for ASR models.
- Visualization tools for SHAP values over spectrograms and waveforms.
- Metrics to evaluate SHAP's effectiveness in identifying speech-relevant regions.
- Correlation analysis between SHAP values, noise, and WER.

## Current Status

- SHAP value computation integrated with Wav2Vec2 ASR models.
- Interactive visualization tools for SHAP values and spectrograms.
- Evaluation framework for SHAP effectiveness using metrics η_raw and WER as described in the paper.
- Controlled test set generation with clean and noisy audio samples.
- Initial results show promising correlations between SHAP values and speech-relevant regions.

## Requirements

To run this project, install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage
### SHAP Value Computation  
To compute SHAP values for a test set:

```bash
python shap_calculation.py
```

This script:

1. Loads a pre-trained Wav2Vec2 model.
2. Creates a controlled test set with clean and noisy audio samples.
3. Computes SHAP values for each sample.
4. Saves the results in the data/ directory.

### Visualization  
To visualize SHAP values interactively:
```bash
python visualization.py
```
This script:

1. Loads audio and SHAP values from the data/ directory.
2. Displays an interactive spectrogram with SHAP overlays.
3. Allows users to explore SHAP-weighted spectrograms for individual characters.

### Evaluation  
To evaluate SHAP effectiveness using η_raw and WER:
```bash
python nraw_vs_wer.py
```
This script:

1. Calculates the η_raw score for each audio sample.
2. Computes the WER for the model's transcription.
3. Generates a scatter plot showing the correlation between η_raw and WER.

### Systematic Evaluation  
To only compute η_raw score for a selected audio run:
```bash
python calculate_metric.py
```
This script:

1. Lets you choose a previously computed set of audio file+shap values
1. Calculates η_raw for selected audio sample.

## Results
The project demonstrates that SHAP values can effectively highlight speech-relevant regions in audio, even under noisy conditions. The correlation between SHAP values and WER provides insights into model behavior and areas for improvement.

## License
This project is licensed under the MIT License. See the LICENSE file for details. 