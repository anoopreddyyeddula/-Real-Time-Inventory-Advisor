import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib
import os

os.makedirs('models', exist_ok=True)
os.makedirs('data', exist_ok=True)

data = pd.read_csv('data/sales_data_large.csv')

product_models = {}

for product_id, group in data.groupby('Product_ID'):
    group = group.sort_values('Date')
    X = group[['Units_Sold']].shift(1).fillna(0)
    y = group['Units_Sold']
    
    model = LinearRegression()
    model.fit(X, y)
    
    product_models[product_id] = model

joblib.dump(product_models, 'models/inventory_forecaster.pkl')

print(f"✅ Trained and saved models for {len(product_models)} products!")