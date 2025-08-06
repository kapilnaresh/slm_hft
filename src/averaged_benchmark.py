import yaml
from inference import RiskSignalDetector
import numpy as np
import time
from sklearn.metrics import accuracy_score, f1_score
from preprocessor import Preprocessor
import os
import psutil
import pandas as pd
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

class Benchmark:
    def __init__(self, config_path):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        self.results = []

    def benchmark_latency(self, model_path, test_texts, num_runs=100):
        predictor = RiskSignalDetector(model_path, 'config/config.yaml')
        for _ in range(10):
            predictor.prediction_for_single_text(test_texts[0])
        single_inference_times = []
        for _ in range(num_runs):
            text = np.random.choice(test_texts)
            start = time.perf_counter()
            predictor.prediction_for_single_text(text)
            end = time.perf_counter()
            single_inference_times.append((end - start) * 1000)
        batch_sizes = self.config["benchmarking"]["batch_sizes"]
        batch_inference_results = {}
        for batch_size in batch_sizes:
            batch_texts = (test_texts * (batch_size // len(test_texts) + 1))[:batch_size]
            batch_inference_times = []
            for _ in range(10):
                start = time.perf_counter()
                predictor.prediction_for_batch(batch_texts, batch_size=batch_size)
                time_taken = (time.perf_counter() - start) * 1000
                batch_inference_times.append(time_taken / batch_size)
            batch_inference_results[batch_size] = {
                "mean_latency": np.mean(batch_inference_times),
                "std latency": np.std(batch_inference_times),
                "p95_latency": np.percentile(batch_inference_times, 90)
            }
        return {
            'model_path': model_path,
            'single_inference': {
                'mean_latency': np.mean(single_inference_times),
                'std_latency': np.std(single_inference_times),
                'min_latency': np.min(single_inference_times),
                'p95_latency': np.percentile(single_inference_times, 95),
                'p99_latency': np.percentile(single_inference_times, 99)
            },
            'batch_inference': batch_inference_results
        }

    def benchmark_accuracy(self, model_path, test_data):
        predictor = RiskSignalDetector(model_path, 'config/config.yaml')
        predictions = []
        true_values = []
        for _, row in test_data.iterrows():
            result = predictor.prediction_for_single_text(row['text'])
            predictions.append(result['predicted_class'])
            true_values.append(row['risk_label'])
        accuracy = accuracy_score(true_values, predictions)
        f1 = f1_score(true_values, predictions, average='weighted')
        return {
            "model_path": model_path,
            "accuracy": accuracy,
            "f1_score": f1
        }

    def benchmark_memory(self, model_path):
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss / 1024 / 1024
        predictor = RiskSignalDetector(model_path, 'config/config.yaml')
        memory_after = process.memory_info().rss / 1024 / 1024
        memory_used = memory_after - memory_before
        num_params = sum(p.numel() for p in predictor.model.parameters())
        model_size_mb = sum(p.numel() * p.element_size() for p in predictor.model.parameters()) / 1024 / 1024
        return {
            'model_path': model_path,
            'memory_usage_mb': memory_used,
            'model_size_mb': model_size_mb,
            'num_parameters': num_params
        }

    def run_all_benchmarks(self, model_paths, test_texts, test_data, num_runs=1):
        all_results = []
        for run in range(num_runs):
            print(f"Benchmark run {run+1}/{num_runs}")
            results = []
            for model_path in model_paths:
                print(f"Benchmarking for {model_path}...")
                latency_results = self.benchmark_latency(model_path, test_texts)
                accuracy_results = self.benchmark_accuracy(model_path, test_data)
                memory_results = self.benchmark_memory(model_path)
                combined_results = {
                    'model_path': model_path,
                    'model_name': model_path.split('/')[-1],
                    **latency_results['single_inference'],
                    **accuracy_results,
                    **memory_results
                }
                results.append(combined_results)
            all_results.extend(results)
        self.results = all_results
        return all_results

    def aggregate_results(self, results, output_path_csv='averaged_benchmark_results.csv', output_path_png='averaged_benchmark_results.png'):
        df = pd.DataFrame(results)
        agg_df = df.groupby('model_name').mean(numeric_only=True).reset_index()
        for col in ['model_path']:
            if col in df.columns:
                agg_df[col] = df.groupby('model_name')[col].first().values
        agg_df.to_csv(output_path_csv, index=False)
        self.create_visualizations(agg_df, output_path_png)
        print("\n=== Aggregated Benchmark Summary ===")
        print(agg_df[['model_name', 'mean_latency', 'accuracy', 'f1_score', 'memory_usage_mb', 'num_parameters']].to_string(index=False))
        return agg_df

    def create_visualizations(self, df: pd.DataFrame, output_path_png='averaged_benchmark_results.png'):
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes[0, 0].scatter(df['mean_latency'], df['accuracy'], s=df['num_parameters']/1e6)
        axes[0, 0].set_xlabel('Mean Latency (ms)')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].set_title('Latency vs Accuracy (bubble size = params)')
        axes[0,0].set_ylim(0.0,1.0)
        for i, txt in enumerate(df['model_name']):
            axes[0, 0].annotate(txt, (df['mean_latency'].iloc[i], df['accuracy'].iloc[i]))
        axes[0, 1].bar(df['model_name'], df['memory_usage_mb'])
        axes[0, 1].set_xlabel('Model')
        axes[0, 1].set_ylabel('Memory Usage (MB)')
        axes[0, 1].set_title('Memory Usage by Model')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[1, 0].bar(df['model_name'], df['mean_latency'], yerr=df['std_latency'])
        axes[1, 0].set_xlabel('Model')
        axes[1, 0].set_ylabel('Mean Latency (ms)')
        axes[1, 0].set_title('Inference Latency by Model')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 1].bar(df['model_name'], df['f1_score'])
        axes[1, 1].set_xlabel('Model')
        axes[1, 1].set_ylabel('F1 Score')
        axes[1, 1].set_title('F1 Score by Model')
        axes[1, 1].tick_params(axis='x', rotation=45)
        plt.tight_layout()
        plt.savefig(output_path_png, dpi=300, bbox_inches='tight')
        plt.close(fig)

def driver():
    test_texts = [
        "Apple reports strong quarterly earnings beating expectations",
        "Market volatility increases as tech stocks decline",
        "SEC launches investigation into company practices",
        "Regular dividend payment announced by Microsoft"
    ] * 25  # 100 test texts
    processer = Preprocessor('config/config.yaml')
    test_data = processer.preprocess(processer.load_data()[1])
    model_paths = [
        'models/distilbert',
        'models/tinybert'  # If available
    ]
    benchmark = Benchmark('config/config.yaml')
    all_results = benchmark.run_all_benchmarks(model_paths, test_texts, test_data, num_runs=1000)
    benchmark.aggregate_results(all_results, output_path_csv='averaged_benchmark_results.csv', output_path_png='averaged_benchmark_results.png')

if __name__ == "__main__":
    driver()
