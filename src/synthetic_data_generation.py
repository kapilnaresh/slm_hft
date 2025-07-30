import random
import pandas as pd
from datetime import datetime, timedelta

#A class that generates synthetic data to train the models
class SyntheticDataGenerator:
    def __init__(self):
        self.templates = {
            "NO_RISK": [
                "{company} reports quarterly earnings meeting analyst expectations",
                "{company} announces regular dividend payment of ${amount}",
                "{company} schedules annual shareholder meeting",
                "{company} releases routine product update",
                "{company} appoints new board member"
            ],
            "MARKET_RISK": [
                "Market volatility spikes as {company} drops {percent}% in heavy trading",
                "Sector rotation affects {company} amid broader market uncertainty",
                "Interest rate concerns impact {company} valuation",
                "{company} falls with tech sector on growth fears",
                "Market correction hits {company} along with major indices"
            ],
            "COMPANY_RISK": [
                "{company} faces investigation by regulatory authorities",
                "CEO of {company} announces unexpected departure",
                "{company} reports earnings miss of {percent}%",
                "{company} faces cybersecurity breach affecting operations",
                "{company} announces major restructuring and layoffs"
            ],
            "REGULATORY_RISK": [
                "{company} receives regulatory fine of ${amount} million",
                "New compliance requirements could impact {company} operations",
                "{company} faces antitrust investigation",
                "Policy changes may affect {company} business model",
                "{company} under scrutiny for data privacy practices"
            ]
        }
        #list of companies being used
        self.companies = ['Apple', 'Google', 'Microsoft', 'Tesla', 'NVIDIA', 'Meta', 'Amazon']
        self.tickers = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA', 'META', 'AMZN']
    
    def generate_synthetic_data(self, num_samples=1000):
        synthetic_data = []
        for i in range(num_samples):
            #using random company and news template
            category = random.choice(list(self.templates.keys()))
            template = random.choice(self.templates[category])
            company_idx = random.randint(0,len(self.companies) - 1)
            company = self.companies[company_idx]
            ticker = self.tickers[company_idx]
            #generating synthetic text
            text = template.format(company=company,
                                   percent=random.randint(1,15),
                                   amount=random.randint(5,50))
            random_time = datetime.now() - timedelta(days=random.randint(0,365))
            synthetic_data.append({
                'ticker': ticker,
                'title': text,
                'summary': text,
                'published': random_time,
                'source': 'synthetic',
                'risk_category': category,
                'risk_label': list(self.templates.keys()).index(category)
            })
        return pd.DataFrame(synthetic_data)

def driver():
    data_generator = SyntheticDataGenerator()
    synthetic_data = data_generator.generate_synthetic_data(1000)
    synthetic_data.to_csv('data/synthetic_data.csv', index=False)
    print("Successfully generated synthetic data")

if __name__ == "__main__":
    driver()
