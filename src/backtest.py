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
    
    #decide how much to allocate depending on risk
    def allocation_size(self,ticker,signal):
        base_allocation = 0.1
        if signal['risk_type'] == 'MARKET_RISK':
            base_allocation *= 0.5
        elif signal['risk_type'] == 'COMPANY_RISK':
            base_allocation = 0.0
        return base_allocation
    

    #execute trades
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

    #calculate total value of portfolio
    def calculate_portfolio(self,prices,date):
        capital = self.capital
        stock_value = 0
        for ticker,shares in self.positions.items():
            if ticker in prices:
                stock_value += shares * prices[ticker]
        total_value = capital + stock_value
        self.portfolio_values.append({
            'date': date,
            'cash': capital,
            'stocks_value': stock_value,
            'total_value': total_value
        })
        return total_value
    
    def run_backtest(self,market_data,risk_signals):
        tickers = market_data.columns.tolist()
        for i,date in enumerate(market_data.index):
            current_prices = market_data.loc[date]
            day_signals = risk_signals[risk_signals['date'] == date]
            if(len(day_signals) > 0):
                for _,signal in day_signals.iterrows():
                    ticker = signal['ticker']
                    risk_type = signal['risk_type']
                    if risk_type == 'COMPANY_RISK':
                        if ticker in self.positions and self.positions[ticker] > 0:
                            self.trade(ticker,'SELL', self.positions[ticker],current_prices[ticker],date)
                    elif risk_type == 'MARKET_RISK':
                        for ticker_idx in list(self.positions.keys()):
                            if self.positions[ticker_idx] > 0:
                                self.trade(ticker_idx, 'SELL', self.positions[ticker_idx] * 0.5,current_prices[ticker_idx,date])

            if i % 30 == 0:
                target_per_stock = (self.capital + sum(self.positions.get(t,0) * current_prices[t] for t in tickers))
                for ticker in tickers:
                    target_shares = target_per_stock / current_prices[ticker]
                    current_shares = self.positions.get(ticker,0)
                    if(target_shares > current_shares * 1.2):
                        buy_shares = min(target_shares-current_shares,self.capital/current_prices[ticker] * 0.1)
                        if(buy_shares > 0):
                            self.trade(ticker,'BUY', buy_shares, current_prices[ticker],date)
        
        portfolio_df = pd.DataFrame(self.portfolio_values)
        portfolio_df['returns'] = portfolio_df['total_value'].pct_change()

        benchmark_return = market_data.mean(axis=1).pct_change()
        results = self.calculate_metrics(portfolio_df,benchmark_return)
        results['trades'] = pd.DataFrame(self.trades)
        results['portfolio_values'] = portfolio_df
        return results