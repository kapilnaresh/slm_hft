import yfinance as yf
import pandas as pd
import numpy as np



class Backtester:
    def __init__(self, initial_capital):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.positions = {}
        self.trades = []
        self.portfolio_values = []
        self.risk_signals = []
    
    def load_data(self, tickers,start_date,end_date):
        data = {}
        for ticker in tickers:
            data = yf.download(ticker,start=start_date,end=end_date,auto_adjust=False)
            adj = data['Adj Close']
            if isinstance(adj,pd.DataFrame):
                adj = adj.squeeze()
            data[ticker] = adj
        return pd.DataFrame(data).dropna()

    def simulate_signals(self,dates,tickers):
        np.random.seed(42)
        signals = []
        for date in dates:
            for ticker in tickers:
                if(np.random.random() < 0.05):
                    risk_type = np.random.choice(['MARKET_RISK','COMPANY_RISK'], p=[0.3,0.7])
                    signals.append({
                        'date': date,
                        'ticker': ticker,
                        'risk_type': risk_type,
                        'confidence': np.random.uniform(0.7,0.95)
                    })
        return pd.DataFrame(signals)