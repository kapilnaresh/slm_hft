import yaml
from inference import RiskSignalDetector
import numpy as np
import time
from sklearn.metrics import accuracy_score,f1_score,classification_report
from preprocessor import Preprocessor
import os
import psutil

class Benchmark:
    def __init__(self,config_path):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        self.results = []
    
    #benchmarking the latency of models
    def benchmark_latency(self,model_path,test_texts,num_runs=100):
        predictor = RiskSignalDetector(model_path,'config/config.yaml')

        #warmup steps
        for _ in range(10):
            predictor.prediction_for_single_text(test_texts[0])
        
        #single text inference benchmark
        single_inference_times = []
        for _ in range(num_runs):
            text = np.random.choice(test_texts)
            start = time.perf_counter()
            predictor.prediction_for_single_text(text)
            end = time.perf_counter()
            single_inference_times.append((end-start)*1000)
        
        #batch text inference benchmark
        batch_sizes = self.config["benchmarking"]["batch_sizes"]
        batch_inference_results = {}
        for batch_size in batch_sizes():
            batch_texts = (test_texts * (batch_size // len(test_texts) + 1))[:batch_size]
            batch_inference_times = []
            for _ in range(10):
                start = time.perf_counter()
                predictor.prediction_for_batch(batch_texts, batch_size=batch_size)
                time_taken = (time.perf_counter() - start) * 1000
                batch_inference_times.append(time_taken / batch_size) # time per text in sample
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
    
    #benchmarking the accuracy/other metrics of the model
    def benchmark_accuracy(self, model_path, test_data):
        predictor = RiskSignalDetector(model_path,'config/config.yaml')
        predictions = []
        true_values = []
        for _,row in test_data.iterrows():
            result = predictor.prediction_for_single_text(row['text'])
            predictions.append(result['predicted_class'])
            true_values.append(row['risk_label'])
        accuracy = accuracy_score(true_values,predictions)
        f1 = f1_score(true_values,predictions)
        return {
            "model_path": model_path,
            "accuracy": accuracy,
            "f1_score": f1,
            'classification_report': classification_report(true_values,predictions)
        }
    
    #benchmarking memory used by each model
    def benchmark_memory(self,model_path):
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss / 1024 / 1024 #get memory used in MB
        predictor = RiskSignalDetector(model_path,'config/config.yaml')
        memory_after = process.memory_info().rss / 1024 / 1024
        memory_used = memory_after - memory_before
        #get num of parameters of the model
        num_params = sum(p.numel() for p in predictor.model.parameters())
        model_size_mb = sum(p.numel() * p.element_size() for p in predictor.model.parameters()) / 1024 / 1024
        return {
            'model_path': model_path,
            'memory_usage_mb': memory_used,
            'model_size_mb': model_size_mb,
            'num_parameters': num_params
        }

    
