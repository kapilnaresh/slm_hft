import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import finnhub
from inference import RiskSignalDetector
import datetime

class Backtester:
    def __init__(self, initial_capital, start_date, end_date):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.positions = {}
        self.trades = []
        self.portfolio_values = []
        self.risk_signals = []
        self.start_date = start_date
        self.end_date = end_date
    
    #load stock data from yahoo for specified dates to test
    def load_data(self, tickers):
        data = {}
        for ticker in tickers:
            stock_data = yf.download(ticker,start=self.start_date,end=self.end_date,auto_adjust=False)
            adj = stock_data['Adj Close']
            if isinstance(adj,pd.DataFrame):
                adj = adj.squeeze()
            data[ticker] = adj
        return pd.DataFrame(data).dropna()
    
    def load_news(self, tickers):
        finnhub_client = finnhub.Client(api_key=os.getenv("FINHUB_API"))
        overall_data = {}
        for ticker in tickers:
            company_news = finnhub_client.company_news(ticker,self.start_date,self.end_date)
            overall_data[ticker] = company_news
        return overall_data
    
    def get_signals(self,tickers,model_path):
        predictor = RiskSignalDetector(model_path, 'config/config.yaml')
        news_data = self.load_news(tickers)
        signals = []
        for ticker in tickers:
            company_news = news_data[ticker]
            for news_item in company_news:
                text = news_item["headline"] + " " + news_item["summary"]
                date = datetime.datetime.fromtimestamp(news_item["datetime"])
                prediction = predictor.prediction_for_single_text(text)
                signals.append({
                    "date": date,
                    "ticker": ticker,
                    "risk_type": prediction["predicted_risk"],
                    "confidence": f"{max(prediction['probabilities']):.3f}"
                })
        return pd.DataFrame(signals)
                
    #simulate news/signals of market risk or company risk randomly
    '''
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
    '''
    

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
                            self.trade(ticker,'SELL', self.positions[ticker] * 0.8,current_prices[ticker],date)
                    elif risk_type == 'MARKET_RISK':
                        for ticker_idx in list(self.positions.keys()):
                            if self.positions[ticker_idx] > 0:
                                self.trade(ticker_idx, 'SELL', self.positions[ticker_idx] * 0.5,current_prices[ticker_idx],date)
                    elif risk_type == 'REGULATORY_RISK':
                        if ticker in self.positions and self.positions[ticker] > 0:
                            self.trade(ticker,'SELL', self.positions[ticker] * 0.6,current_prices[ticker],date)

            if i % 20 == 0:
                target_per_stock = (self.capital + sum(self.positions.get(t,0) * current_prices[t] for t in tickers))
                for ticker in tickers:
                    current_value = self.positions.get(ticker, 0) * current_prices[ticker]
                    target_shares = target_per_stock / current_prices[ticker]
                    current_shares = self.positions.get(ticker,0)
                    if(target_shares > current_shares * 1.1):
                        buy_shares = min(target_shares-current_shares,self.capital/current_prices[ticker] * 0.1)
                        if(buy_shares > 0):
                            self.trade(ticker,'BUY', buy_shares, current_prices[ticker],date)
            portfolio_value = self.calculate_portfolio(current_prices, date)
        portfolio_df = pd.DataFrame(self.portfolio_values)
        portfolio_df['returns'] = portfolio_df['total_value'].pct_change()

        benchmark_return = market_data.mean(axis=1).pct_change()
        results = self.compute_metrics(portfolio_df,benchmark_return)
        results['trades'] = pd.DataFrame(self.trades)
        results['portfolio_values'] = portfolio_df
        return results
    

    def compute_metrics(self,portfolio_df, benchmark_returns):
        portfolio_returns = portfolio_df['returns'].dropna()
        total_return = (portfolio_df['total_value'].iloc[-1] / self.initial_capital - 1) * 100
        days = len(portfolio_returns)
        annualized_return = ((portfolio_df['total_value'].iloc[-1] / self.initial_capital) ** (252/days) - 1) * 100
        volatility = portfolio_returns.std() * np.sqrt(252) * 100
        # Sharpe ratio (assume 2% risk-free rate)
        risk_free_rate = 0.02
        sharpe_ratio = (annualized_return/100 - risk_free_rate) / (volatility/100)
                # Maximum drawdown
        rolling_max = portfolio_df['total_value'].expanding().max()
        drawdown = (portfolio_df['total_value'] - rolling_max) / rolling_max
        max_drawdown = drawdown.min() * 100    
        # Win rate
        win_rate = (portfolio_returns > 0).mean() * 100
        
        # Benchmark comparison
        benchmark_total_return = ((1 + benchmark_returns).prod() - 1) * 100
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'benchmark_return': benchmark_total_return,
            'alpha': total_return - benchmark_total_return,
            'num_trades': len(self.trades)
        }


    def plot_results(self, results):
        """Plot backtest results"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        portfolio_df = results['portfolio_values']
        # Portfolio value over time
        axes[0, 0].plot(portfolio_df['date'], portfolio_df['total_value'], label='Strategy')
        axes[0, 0].plot(portfolio_df['date'], 
                       [self.initial_capital * (1 + results['benchmark_return']/100 * i/len(portfolio_df)) 
                        for i in range(len(portfolio_df))], 
                       label='Benchmark', linestyle='--')
        axes[0, 0].set_title('Portfolio Value Over Time')
        axes[0, 0].set_ylabel('Portfolio Value ($)')
        axes[0, 0].legend()
        # Drawdown
        rolling_max = portfolio_df['total_value'].expanding().max()
        drawdown = (portfolio_df['total_value'] - rolling_max) / rolling_max * 100
        axes[0, 1].fill_between(portfolio_df['date'], drawdown, 0, alpha=0.3, color='red')
        axes[0, 1].set_title('Drawdown')
        axes[0, 1].set_ylabel('Drawdown (%)')
        # Returns distribution
        returns = portfolio_df['returns'].dropna()
        axes[1, 0].hist(returns, bins=50, alpha=0.7)
        axes[1, 0].set_title('Returns Distribution')
        axes[1, 0].set_xlabel('Daily Returns')
        axes[1, 0].set_ylabel('Frequency')
        # Performance metrics
        metrics_text = f"""
        Total Return: {results['total_return']:.2f}%
        Annualized Return: {results['annualized_return']:.2f}%
        Volatility: {results['volatility']:.2f}%
        Sharpe Ratio: {results['sharpe_ratio']:.2f}
        Max Drawdown: {results['max_drawdown']:.2f}%
        Alpha: {results['alpha']:.2f}%
        Number of Trades: {results['num_trades']}
        """
        axes[1, 1].text(0.1, 0.9, metrics_text, transform=axes[1, 1].transAxes, 
                        fontsize=10, verticalalignment='top', fontfamily='monospace')
        axes[1, 1].set_title('Performance Metrics')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig('backtest_results.png', dpi=300, bbox_inches='tight')
        plt.show()

def driver():
    # Initialize backtester
    backtester = Backtester(initial_capital=100000, start_date = '2024-08-30',end_date = '2025-06-30')
    
    # Load market data
    tickers = ['AAPL', 'GOOGL', 'MSFT', 'NVDA', 'TSLA']
    
    
    market_data = backtester.load_data(tickers)
    
    # Simulate risk signals
    risk_signals = backtester.get_signals(tickers,'models/distilbert')
    
    print(f"Loaded {len(market_data)} days of market data")
    print(f"Generated {len(risk_signals)} risk signals")
    
    # Run backtest
    results = backtester.run_backtest(market_data, risk_signals)
    
    # Display results
    print("\n=== Backtest Results ===")
    for metric, value in results.items():
        if isinstance(value, (int, float)):
            print(f"{metric}: {value:.2f}")
    
    # Plot results
    backtester.plot_results(results)
    with open("trades.txt", "w") as f:
        f.write(f"{backtester.trades}")
if __name__ == "__main__":
    driver()