import yfinance as yf
import yaml
import pandas as pd
from datetime import datetime
import requests

# class to collect real data to train models and use for inference
class DataCollector:
    #setting up config
    def __init__(self,config_path):
        with open(config_path, "r") as file:
            self.config = yaml.safe_load(file)
    
    #getting news through yahoo finance API
    def collect_yahoo_news(self, tickers):
        news = []
        for ticker in tickers:
            stock = yf.Ticker(ticker)
            stock_news = stock.news

            for article in stock_news:
                news.append(
                    {'ticker': ticker,
                     'title': article['content']['title'],
                     'summary': article['content']['summary'],
                     'published' : article['content']['pubDate'],
                     'source':'yahoo'
                    }
                )
        return pd.DataFrame(news)
    
    #getting filings of companies from SEC EDGAR API
    def collect_sec_filings(self,tickers):
        filings = []
        for ticker in tickers:
            url = f"https://data.sec.gov/submissions/CIK{self.cik_mapper(ticker)}.json"
            headers = {'User-Agent': "Research"}

            try:
                response = requests.get(url,headers=headers)
                if response.status_code == 200:
                    data = response.json()
                    latest_filings = data['filings']['recent']
                    for i in range(min(15,len(latest_filings['form']))):
                        filings.append({
                            'ticker':ticker,
                            'title': f"{latest_filings['form'][i]} Filing",
                            'summary': latest_filings['primaryDocument'][i],
                            'published': datetime.strptime(latest_filings['filingDate'][i], '%Y-%m-%d'),
                            'source': 'sec'
                        })
            except Exception as e:
                print(f"Error obtaining SEC details in data_collection.py due to {e}\n")
        return pd.DataFrame(filings)

    #maps ticker to cik number used in SEC filings
    def cik_mapper(self, ticker):
        cik_mapping = {
            'AAPL': '0000320193',
            'GOOGL': '0001652044', 
            'MSFT': '0000789019',
            'TSLA': '0001318605',
            'NVDA': '0001045810'
        }
        return cik_mapping.get(ticker, '0000000000')
    
    #main function that runs everything
def driver():
    data_collector = DataCollector('config/config.yaml')
    tickers = data_collector.config['data']['tickers']
    sec_filings = data_collector.collect_sec_filings(tickers)
    yahoo_news = data_collector.collect_yahoo_news(tickers)
    final_data = pd.concat([yahoo_news,sec_filings], ignore_index=True)
    final_data.to_csv('data/real_financial_news.csv', index=False)
    print("Successfully collected financial data")
    
if __name__ == "__main__":
    driver()