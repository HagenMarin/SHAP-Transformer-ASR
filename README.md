# SHAP-Transformer-ASR
Exploring SHAP for interpreting Transformer-based ASR models. Provides explainability via Shapley values to improve transparency in speech recognition. Includes code, benchmarks, and analysis for debugging &amp; trust in high-stakes applications.

## Project Plan

### 1. Data Collection and Preparation
- Utilize LibriSpeech dataset
- Create diverse test set including:
  - Clean speech samples
  - Noisy speech samples with specific categories:
    - Stationary noise (white noise, pink noise)
    - Non-stationary noise (babble, music)
    - Environmental noise (traffic, wind)
  - Different accents and speakers
  - Various speech lengths
- Add metadata for each sample (speaker ID, accent, noise level, etc.)
- Generate reference masks:
  - Ideal Binary Mask (IBM) for each sample
  - Ideal Ratio Mask (IRM) for each sample

### 2. Model Analysis Framework
- Systematic evaluation pipeline
- Key components:
  - Temporal importance analysis
  - Frequency importance analysis
  - Word-level importance mapping
  - Confidence score integration
  - Statistical analysis tools
  - Human annotation comparison
- SHAP Implementation Strategy:
  - Input features (mel spectrograms/MFCCs)
  - Attention weights in Transformer layers
  - Output probabilities for each token
  - KernelSHAP for feature-level explanations
  - Attention visualization tools

### 3. Visualization and Analysis Components
- Enhanced visualizations:
  - Word-level alignment
  - Confidence scores
  - Reference text integration
  - Interactive elements
- Comparative visualizations:
  - SHAP vs. attention weights
  - SHAP vs. model confidence
  - SHAP vs. human annotations
- SHAP-specific visualizations:
  - Heatmaps of SHAP values over spectrograms
  - Attention patterns with SHAP overlays
  - Comparative visualizations between model layers
  - Interactive analysis tools

### 4. Evaluation Metrics
- Quantitative metrics:
  - Temporal consistency
  - Frequency band importance
  - Word-level attribution accuracy
  - Correlation with model confidence
  - Word Error Rate (WER) correlation with SHAP values
  - Feature importance ranking
  - Attention pattern analysis
  - Computational efficiency metrics
  - Mask-based evaluation:
    - Correlation between SHAP values and IBM/IRM masks
    - Accuracy of speech/noise region identification
    - Consistency across different noise types
    - Impact on ASR performance
- Qualitative evaluation:
  - Human annotation guidelines
  - Explanation quality assessment
  - Use case specific metrics

### 5. Key Analysis Areas
- SHAP value correlation analysis:
  - Phoneme boundaries
  - Word boundaries
  - Speaker characteristics
  - Background noise
  - Mask-based analysis:
    - Comparison with IBM/IRM reference masks
    - Noise type-specific patterns
    - Speech enhancement effectiveness
- Comparative analysis across:
  - Different speaker accents
  - Various speaking rates
  - Different noise conditions
  - Multiple model architectures
  - Different mask types (IBM vs IRM)

### 6. Implementation Phases
1. Basic Framework
   - Enhanced SHAP visualization
   - Basic evaluation metrics
   - Core analysis tools
   - SHAP integration with ASR pipeline

2. Advanced Analysis
   - Word-level analysis
   - Comparative methods
   - Evaluation pipeline
   - Attention pattern analysis

3. Comprehensive Evaluation
   - Human evaluation integration
   - Statistical analysis
   - Documentation and examples
   - Cross-validation across datasets

### 7. Validation Framework
- Ablation studies
- Baseline method comparisons:
  - LIME
  - Integrated Gradients
  - IBM/IRM-based evaluation
- Human evaluation of explanations
- Cross-validation across datasets
- Mask-based validation:
  - Comparison with ideal masks
  - Noise type-specific validation
  - Speech enhancement metrics

### 8. Expected Challenges and Solutions
- Computational complexity of SHAP
  - Solution: Use approximation methods and GPU acceleration
- Interpreting attention patterns
  - Solution: Develop layer-specific visualization tools
- Scalability to long audio sequences
  - Solution: Implement efficient sampling strategies

### 9. Documentation and Reporting
- Detailed documentation:
  - Methodology
  - Evaluation metrics
  - Visualization guidelines
- Example reports:
  - Sample analysis
  - Comparative results
  - Best practices
- Technical documentation
- Visualization examples
- Comparative analysis

## Current Status
- Basic SHAP visualization implemented
- Mel spectrogram with SHAP value overlay
- Initial model integration with Wav2Vec2
- IBM/IRM mask generation pipeline
- Basic implementation of SHAP for ASR model is working
- Custom handlers for LayerNorm, SiLU, GroupNorm, and GLU modules are implemented
- Currently using simplified handlers that delegate to SHAP's core `linear_1d` and `nonlinear_1d` functions
- Additivity error is still above the desired tolerance (0.01)
- Model output and SHAP values show some discrepancy in their sums

## Next Steps
1. Enhance visualization with word-level alignment
2. Implement basic evaluation metrics
3. Add support for different model architectures
4. Create comparison framework
5. Implement SHAP integration for attention layers
6. Develop comprehensive evaluation pipeline
7. Integrate IBM/IRM-based evaluation
8. Implement noise-specific analysis tools
9. Fix custom handlers for PyTorch modules:
   - Improve gradient handling in custom handlers
   - Ensure proper attribute handling for x and y tensors
   - Reduce additivity error to below 0.01
   - Consider implementing more sophisticated gradient computation methods
10. Improve visualization:
    - Add more detailed plots showing SHAP values over time
    - Include confidence intervals for SHAP values
    - Add interactive visualization options
11. Performance optimization:
    - Reduce memory usage during SHAP computation
    - Optimize background dataset creation
    - Consider implementing batched processing
12. Documentation:
    - Add detailed API documentation
    - Include examples for different use cases
    - Document the model architecture and SHAP implementation details

## Requirements
See `requirements.txt` for detailed dependencies.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```python
python test_shap_audio.py
```

This will:
1. Load a sample audio file
2. Process it through the ASR model
3. Compute SHAP values
4. Generate visualizations
5. Save the results

## License

MIT
