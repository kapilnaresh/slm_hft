import pandas as pd

# Load the CSV file
file_path = "data/unseen_categorized_stock_news.csv"  # Replace with your actual file path
df = pd.read_csv(file_path)

# Define the mapping
risk_mapping = {
    "NO_RISK": 0,
    "MARKET_RISK": 1,
    "COMPANY_RISK": 2,
    "REGULATORY_RISK": 3,
    "POSITIVE": 4
}

# Add the numeric risk code column
df["risk_code"] = df["risk_label"].map(risk_mapping)

# Save the updated CSV
output_path = "data/unseen_categorized_stock_news.csv"
df.to_csv(output_path, index=False)

print(f"Updated file saved to {output_path}")
