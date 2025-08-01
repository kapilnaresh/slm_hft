import yaml
import torch
import numpy as np
from sklearn.metrics import accuracy_score,f1_score
from preprocessor import Preprocessor
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, EarlyStoppingCallback
import warnings

warnings.filterwarnings("ignore")

class ModelTrainer:
    def __init__(self,config_path):
        with open(config_path, "r") as file:
            self.config = yaml.safe_load(file)
        if(torch.backends.mps.is_available()):
            self.device = "mps"
        else:
            self.device = "cpu"
    
    #compute metrics for evaluation
    def compute(self,evaluation_predictions):
        predictions,labels = evaluation_predictions
        predictions = np.argmax(predictions, axis=1)
        return {
            'accuracy': accuracy_score(labels,predictions),
            'f1': f1_score(labels,predictions, average='weighted')
        }
    
    def train_model(self,model,output_dir):
        preprocessor = Preprocessor('config/config.yaml')
        train_dataset,val_dataset,test_dataset,tokenizer = preprocessor.train_test_val_split(model)
        no_labels = len(self.config['risk_categories'])
        training_model = AutoModelForSequenceClassification.from_pretrained(model,num_labels=no_labels)

        #using TrainingArguments class to specify parameters
        train_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=self.config['training']['epochs'],
            per_device_train_batch_size=self.config['models']['distilbert']['batch_size'],
            per_device_eval_batch_size=self.config['models']['distilbert']['batch_size'],
            warmup_steps=self.config['training']['warmup_steps'],
            weight_decay=0.01,
            logging_dir=f'{output_dir}/logs',
            logging_steps=100,
            eval_steps=self.config['training']['eval_steps'],
            eval_strategy="steps",
            save_strategy="steps",
            save_steps=self.config['training']['eval_steps'],
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
        )

        trainer = Trainer(model=training_model,
                          args=train_args,
                          train_dataset=train_dataset,
                          eval_dataset=val_dataset,
                          compute_metrics=self.compute,
                          callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
                        )
        print("Training")
        trainer.train()
        results = trainer.evaluate(test_dataset)
        print(f"results: {results}")
        trainer.save_model()
        tokenizer.save_pretrained(output_dir)
        return trainer,results

def driver():
    trainer = ModelTrainer('config/config.yaml')
    #distilbert
    print("Training DistilBERT...")
    distilbert_trainer, distilbert_results = trainer.train_model(
        'distilbert-base-uncased',
        'models/distilbert'
    )
    
    #tinybert
    print("Training TinyBERT...")
    try:
        tinybert_trainer, tinybert_results = trainer.train_model(
            'huawei-noah/TinyBERT_General_4L_312D',
            'models/tinybert'
        )
    except Exception as e:
        print(f"TinyBERT training failed: {e}")

if __name__ == "__main__":
    driver()