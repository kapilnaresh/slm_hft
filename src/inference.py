import yaml
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import time
import numpy as np

#class that classifies the type of risk
class RiskSignalDetector:
    #loading config, model and setting up gpu/cpu
    def __init__(self, model_path, config_path):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        if(torch.backends.mps.is_available()):
            self.device = "mps"
        else:
            self.device = "cpu"
        self.model.to(self.device)
        self.model.eval()
        self.risk_categories = self.config['risk_categories']

    #predicting risk for single text
    def prediction_for_single_text(self,text):
        start_time = time.perf_counter()
        inputs=self.tokenizer(text,return_tensors="pt",truncation=True,
                              padding=True,max_length=512).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilites=torch.nn.functional.softmax(outputs.logits, dim=-1)
            prediction = torch.argmax(probabilites,dim=-1).item()
        end_time = time.perf_counter()
        #converting to ms
        inference_time = (end_time - start_time) * 1000
        return {
            "text": text,
             "predicted_class": prediction,
             "predicted_risk": self.risk_categories[int(prediction)],
             "probabilities": probabilites.cpu().numpy().tolist()[0],
             "inference_time_ms": inference_time
        }

    def prediction_for_batch(self,texts,batch_size=32):
        ret = []
        for i in range(0,len(texts),batch_size):
            subtexts = texts[i:i+batch_size]
            start_time = time.perf_counter()
            inputs=self.tokenizer(subtexts,return_tensors="pt",truncation=True,
                              padding=True,max_length=512).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                probabilites=torch.nn.functional.softmax(outputs.logits, dim=-1)
                predictions = torch.argmax(probabilites,dim=-1)
            end_time = time.perf_counter()
            inference_time = (end_time - start_time) * 1000

            for j,text in enumerate(subtexts):
                ret.append({
            "text": text,
             "predicted_class": predictions[j],
             "predicted_risk": self.risk_categories[int(predictions[j].item())],
             "probabilities": probabilites[j].cpu().numpy().tolist(),
             "batch_inference_time_ms": inference_time / len(subtexts)
        })
        return ret
    
    #approximation of real data
    def predict_for_text_stream(self,text_stream):
        ret = []
        for text in text_stream:
            result = self.prediction_for_single_text(text)
            ret.append(result)
            if(result['inference_time_ms'] > 10):
                print(f"WARNING: inference time exceeds 10ms. Time taken is {result['inference_time_ms']}")
    
def driver():
    predictor = RiskSignalDetector('models/distilbert','config/config.yaml')

    test_texts = [
        "Apple reports strong quarterly earnings beating analyst expectations",
        "Market volatility spikes as Tesla drops 15% in heavy trading",
        "SEC launches investigation into Meta's data privacy practices",
        "Microsoft announces regular dividend payment increase"
    ]

    #single prediction test
    # Single predictions
    print("=== Single Predictions ===")
    for text in test_texts:
        result = predictor.prediction_for_single_text(text)
        print(f"Text: {text}...")
        print(f"Risk: {result['predicted_risk']}")
        print(f"Confidence: {max(result['probabilities']):.3f}")
        print(f"Latency: {result['inference_time_ms']:.2f}ms")
        print("-" * 50)
        # Batch prediction
    print("\n=== Batch Prediction ===")
    batch_results = predictor.prediction_for_batch(test_texts * 10)  # 40 texts
    avg_latency = np.mean([r['batch_inference_time_ms'] for r in batch_results])
    print(f"Average batch latency: {avg_latency:.2f}ms per text")

if __name__ == "__main__":
    driver()


