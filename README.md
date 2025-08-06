# Real-Time Risk Signal Detection Using Small Language Models in High-Frequency Trading


## Overview

This project implements and compares Small Language Models (SLMs) for real-time financial risk detection in high-frequency trading environments. We benchmark **DistilBERT** vs **TinyBERT** to determine optimal speed-accuracy trade-offs for sub-10ms inference requirements.

### Key Research Question
> Can TinyBERT achieve ≥90% of DistilBERT's risk detection accuracy while providing faster inference times for HFT applications?

## Features

- **Sub-10ms Inference**: Optimized for high-frequency trading latency requirements
-  **Comprehensive Benchmarking**: Latency, accuracy, and memory usage comparison
- **Financial Domain Focus**: 6 risk categories tailored for trading applications
- **Backtesting Framework**: Quantitative validation with portfolio performance metrics

## Risk Categories

1. **NO_RISK**: Regular business operations
2. **MARKET_RISK**: Broad market volatility and corrections
3. **COMPANY_RISK**: Company-specific operational issues
4. **REGULATORY_RISK**: Legal investigations and compliance issues
5. **POSITIVE**: Positive news regarding the company

## Project Structure

```
slm-risk-detection/
├── data/
├── src/
│   ├── preprocessing.py   # Data cleaning and tokenization
│   ├── training.py      # Model fine-tuning pipeline
│   ├── inference.py    # Real-time prediction engine
│   ├── benchmark.py        # Performance evaluation suite
│   └── backtest.py         # Financial validation framework for day to day
│   └── backtest.py         # Financial validation framework for minute to minute
├── config/
│   └── config.yaml           # Model and training configuration
├── README.md                  # README file
├── requirements.txt
├── averaged_benchmark_results.csv #csv results of benchmark results averaged over 1000 runs
├── averaged_benchmark_results.png  #plots of results of averaged benchmark
```

## Quick Start


### Installation

```bash
# Clone the repository
git clone https://github.com/kapil-naresh/slm-hft.git
cd slm-gft

# Create virtual environment
conda create -n risk_signals python=3.12
conda activate risk_signals

# Install dependencies
pip install -r requirements.txt
```

### Run Complete Pipeline

```bash

# Or run individual stages
python preprocessor.py      # Preprocess the data
python training.py          # Train the two models
python inference.py         # Test out the inference
python averaged_benchmark.py  #benchmark performance 1000 times
python backtest.py         # Financial validation for day to day basis (less applicable)
python backtest_minute.py  # Financial validation for minute to minute basis more applicable)
```

## Usage Examples

### Quick Model Inference

```python
from src.inference import RiskSignalDetector

# Load trained model
detector = RiskSignalDetector('models/distilbert', 'config/config.yaml')

# Predict risk for news text
result = detector.predict_single(
    "Tesla stock drops 15% following SEC investigation announcement"
)

print(f"Risk: {result['predicted_risk']}")
print(f"Confidence: {max(result['probabilities']):.3f}")
print(f"Latency: {result['inference_time_ms']:.2f}ms")
```

### Batch Processing

```python
news_articles = [
    "Apple reports strong quarterly earnings beating expectations",
    "Market volatility increases amid regulatory uncertainty",
    "Google faces antitrust investigation by DOJ"
]

batch_results = detector.predict_batch(news_articles, batch_size=32)
for result in batch_results:
    print(f"{result['predicted_risk']}: {result['text'][:50]}...")
```

### Custom Training

```python
from src.training import ModelTrainer

trainer = ModelTrainer('config/config.yaml')

# Train custom model
trainer.train_model(
    model_name='distilbert-base-uncased',
    output_dir='models/custom_distilbert'
)
```

## Configuration

Edit `config/config.yaml` to customize:

```yaml
models:
  distilbert:
    name: "distilbert-base-uncased"
    max_length: 512
    batch_size: 16

training:
  epochs: 3
  learning_rate: 2e-5
  warmup_steps: 500

data:
  tickers: ["AAPL", "GOOGL", "MSFT", "TSLA", "NVDA"]
  sample_size: 1000
```

## Benchmark Results

### Model Comparison

| Model | Parameters | Inference Time (ms) | Accuracy | Memory Usage (MB) |
|-------|------------|-------------------|----------|-------------|
| DistilBERT | 66M | ~9.1ms | 74.2% |~7.5 MB
| TinyBERT | 14.5M | ~5.0ms | 67.8% |~2.7 MB

### Performance Visualizations

The benchmarking suite generates:
- **Latency vs Accuracy scatter plot** (bubble size = model parameters)
- **Memory usage comparison** across models  
- **Portfolio performance** over time
- **Risk signal distribution** analysis

## Financial Validation

### Backtesting Results

```
N/A right now
```

## Research Applications

This framework supports research in:
- **Financial NLP**: Domain-specific language model evaluation
- **Real-time ML**: Low-latency inference optimization
- **Quantitative Trading**: Risk signal integration and backtesting
- **Model Compression**: Speed-accuracy trade-off analysis



## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


