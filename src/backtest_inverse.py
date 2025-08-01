# backtest_inverse.py
"""
Inverse strategy: Buy on bad news, sell on positive news, high-frequency backtest.
"""
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import finnhub
from datetime import datetime, timedelta
from inference import RiskSignalDetector
import pytz

class InverseBacktester:
    def __init__(self, initial_capital, start_date, end_date):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.positions = {}
        self.trades = []
        self.portfolio_values = []
        self.start_date = start_date
        self.end_date = end_date

    def load_minute_data(self, tickers):
        eastern = pytz.timezone('US/Eastern')
        data = {}
        for ticker in tickers:
            df = yf.download(ticker, start=self.start_date, end=self.end_date, interval='1m', auto_adjust=False, progress=False)
            if 'Adj Close' in df.columns:
                adj = df['Adj Close']
            else:
                adj = df['Close']
            if isinstance(adj,pd.DataFrame):
                adj = adj.squeeze()
            data[ticker] = adj
        df = pd.DataFrame(data).dropna()
        df.index = pd.to_datetime(df.index)
        df.index = df.index.tz_convert(eastern)
        return df

    def load_news(self, tickers):
        eastern = pytz.timezone('US/Eastern')
        finnhub_client = finnhub.Client(api_key=os.getenv("FINHUB_API"))
        news_events = []
        for ticker in tickers:
            news = finnhub_client.company_news(ticker, self.start_date, self.end_date)
            for item in news:
                dt_utc = datetime.fromtimestamp(item['datetime'], tz=pytz.UTC)
                dt_est = dt_utc.astimezone(eastern)
                news_events.append({
                    'date': dt_est,
                    'ticker': ticker,
                    'headline': item['headline'],
                    'summary': item['summary']
                })
        news_df = pd.DataFrame(news_events)
        news_df = news_df.sort_values('date')
        return news_df

    def get_signals(self, news_df, model_path):
        predictor = RiskSignalDetector(model_path, 'config/config.yaml')
        signals = []
        for _, row in news_df.iterrows():
            text = row['headline'] + ' ' + row['summary']
            prediction = predictor.prediction_for_single_text(text)
            signals.append({
                'date': row['date'],
                'ticker': row['ticker'],
                'risk_type': prediction['predicted_risk'],
                'confidence': max(prediction['probabilities'])
            })
        signals_df = pd.DataFrame(signals)
        signals_df['date'] = signals_df['date'].dt.round('min')
        return signals_df

    def trade(self, ticker, action, size, price, date):
        if action == 'BUY':
            cost = size * price
            if cost <= self.capital:
                self.positions[ticker] = self.positions.get(ticker, 0) + size
                self.capital -= cost
                self.trades.append({'date': date, 'ticker': ticker, 'action': action, 'size': size, 'price': price, 'value': cost})
        elif action == 'SELL':
            if ticker in self.positions and self.positions[ticker] >= size:
                self.positions[ticker] -= size
                self.capital += size * price
                self.trades.append({'date': date, 'ticker': ticker, 'action': action, 'size': size, 'price': price, 'value': size * price})

    def calculate_portfolio(self, prices, date):
        stock_value = sum(self.positions.get(t, 0) * prices.get(t, 0) for t in self.positions)
        total_value = self.capital + stock_value
        self.portfolio_values.append({'date': date, 'cash': self.capital, 'stocks_value': stock_value, 'total_value': total_value})
        return total_value

    def run_backtest(self, market_data, signals):
        tickers = market_data.columns.tolist()
        signals = signals.sort_values('date')
        signal_idx = 0
        signals_list = signals.to_dict('records')
        for i, date in enumerate(market_data.index):
            # Process all signals up to this second
            while signal_idx < len(signals_list) and signals_list[signal_idx]['date'] <= date:
                signal = signals_list[signal_idx]
                ticker = signal['ticker']
                risk_type = signal['risk_type']
                price = market_data.loc[date, ticker] if ticker in market_data.columns else None
                if price is not None:
                    # Inverse logic: Buy on BAD news, sell on POSITIVE news
                    if risk_type in ['COMPANY_RISK', 'MARKET_RISK', 'REGULATORY_RISK', 'NEGATIVE', 'NEGATIVE_NEWS']:
                        shares_to_buy = min(self.capital * 0.05 / price, 10)
                        if shares_to_buy > 0:
                            self.trade(ticker, 'BUY', shares_to_buy, price, date)
                    elif risk_type in ['POSITIVE', 'POSITIVE_NEWS']:
                        if self.positions.get(ticker, 0) > 0:
                            self.trade(ticker, 'SELL', self.positions[ticker], price, date)
                signal_idx += 1

            # New logic: If price is lower than 1 hour ago and most recent news is positive, buy some shares
            for ticker in tickers:
                if i >= 60:
                    price_now = market_data.loc[date, ticker]
                    price_1hr_ago = market_data.iloc[i-60][ticker]
                    if price_now < price_1hr_ago:
                        ticker_signals = signals[(signals['ticker'] == ticker) & (signals['date'] <= date)]
                        if not ticker_signals.empty:
                            last_signal = ticker_signals.iloc[-1]
                            if last_signal['risk_type'] in ['POSITIVE', 'POSITIVE_NEWS']:
                                shares_to_buy = min(self.capital * 0.05 / price_now, 10)
                                if shares_to_buy > 0:
                                    self.trade(ticker, 'BUY', shares_to_buy, price_now, date)

            self.calculate_portfolio(market_data.loc[date], date)
        portfolio_df = pd.DataFrame(self.portfolio_values)
        portfolio_df['returns'] = portfolio_df['total_value'].pct_change()
        return portfolio_df

def main():
    initial_capital = 100000
    tickers = ['AAPL', 'GOOGL', 'MSFT']
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)
    backtester = InverseBacktester(initial_capital, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
    market_data = backtester.load_minute_data(tickers)
    news_df = backtester.load_news(tickers)
    signals = backtester.get_signals(news_df, 'models/distilbert')
    signals.to_csv("singals_inverse.csv")
    portfolio_df = backtester.run_backtest(market_data, signals)
    print(f"Number of trades made: {len(backtester.trades)}")
    plt.plot(portfolio_df['date'], portfolio_df['total_value'])
    plt.title('Inverse Strategy Portfolio Value Over Time (Second-by-Second)')
    plt.xlabel('Time')
    plt.ylabel('Portfolio Value ($)')
    plt.tight_layout()
    plt.savefig('backtest_inverse_results.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    main()
