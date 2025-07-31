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
    
    #load stock data from yahoo for specified dates to test
    def load_data(self, tickers,start_date,end_date):
        data = {}
        for ticker in tickers:
            data = yf.download(ticker,start=start_date,end=end_date,auto_adjust=False)
            adj = data['Adj Close']
            if isinstance(adj,pd.DataFrame):
                adj = adj.squeeze()
            data[ticker] = adj
        return pd.DataFrame(data).dropna()

    #simulate news/signals of market risk or company risk randomly
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
    
    def allocation_size(self,ticker,signal):
        base_allocation = 0.1
        if signal['risk_type'] == 'MARKET_RISK':
            base_allocation *= 0.5
        elif signal['risk_type'] == 'COMPANY_RISK':
            base_allocation = 0.0
        return base_allocation

    def trade(self, ticker, action, size, price, date):
        if action == 'BUY':
            cost = size * price
            if cost <= self.capital:
                self.positions[ticker] = self.positions.get(ticker,0) + size
                self.capital -= cost
                self.trades.append({
                    "date": date,
                    "ticker": ticker,
                    "action": action,
                    "size": size,
                    "price": price,
                    "value": cost
                })
        elif action == "SELL":
            if ticker in self.positions and self.positions[ticker] >= size:
                self.positions[ticker] -= size
                self.capital += size * price
                self.trades.append({
                    "date": date,
                    "ticker": ticker,
                    "action": action,
                    "size": size,
                    "price": price,
                    "value": price * size
                })
        