from torch.utils.data import Dataset
import torch
import yaml
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

class NewsDataset(Dataset):
    def __init__(self,texts,labels,tokenizer,max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)

    #setting up the dataset to be tokenized and setting up tokenizer
    def __getitem__(self,idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])
        encoding = self.tokenizer(text,truncation=True,padding='max_length',max_length = self.max_length, return_tensors='pt')
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

#class where actual preprocessing happens
class Preprocessor:
    def __init__(self,config_path):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        self.risk_categories = self.config['risk_categories']
    
    def load_data(self):
        try:
            synthetic_data = pd.read_csv('data/synthetic_data.csv')
        except FileNotFoundError:
            print("Could not find real financial data in preprocessing.py\n")
        return synthetic_data
    
    #cleaning up titles and summaries 
    def preprocess(self,df):
        df['text'] = df['title'] + ' ' + df['summary']
        df['text'] = df['text'].str.replace(r'[^\w\s]', '',regex=True)
        df['text'] = df['text'].str.replace(r'\s+', '',regex=True)
        df['text'] = df['text'].str.strip()
        df = df[df['text'].str.len() > 10]
        return df
    #create pytorch appropriate datasets for training, validating and testing
    def train_test_val_split(self, model):
        df = self.load_data()
        df = self.preprocess(df)
        train_df, val_test_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df['risk_label'])
        val_df, test_df = train_test_split(val_test_df, test_size=0.5,random_state=42,stratify=val_test_df['risk_label'])
        tokenizer = AutoTokenizer.from_pretrained(model)
        train_dataset = NewsDataset(
            texts=train_df['text'].tolist(),
            labels=train_df['risk_label'].tolist(),
            tokenizer=tokenizer
        )
        val_dataset = NewsDataset(
            texts=val_df['text'].tolist(),
            labels=val_df['risk_label'].tolist(),
            tokenizer=tokenizer
        )
        test_dataset = NewsDataset(
            texts=test_df['text'].tolist(),
            labels=test_df['risk_label'].tolist(),
            tokenizer=tokenizer
        )
        return train_dataset,val_dataset,test_dataset,tokenizer

def driver():
    preprocessor = Preprocessor('config/config.yaml')
    train_dataset,val_dataset,test_dataset,tokenizer = preprocessor.train_test_val_split('distilbert-base-uncased')
    tokenizer.save_pretrained('models/distilbert/tokenizer')
    print("tokenizer saved")

if __name__ == "__main__":
    driver()