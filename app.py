import gradio as gr
import pandas as pd
import joblib
from transformers import pipeline

# Load all ML models
product_models = joblib.load('models/inventory_forecaster.pkl')
llm = pipeline("text2text-generation", model="google/flan-t5-base")

# Function to predict and generate restocking advice
def inventory_advisor(product_id, current_inventory, last_day_sales):
    # Select correct model
    if product_id not in product_models:
        return f"❌ Error: Product ID {product_id} not found in models."

    forecast_model = product_models[product_id]
    future_sales = forecast_model.predict([[last_day_sales]])[0]

    prompt = (f"Current inventory is {current_inventory} units. "
              f"Predicted sales for next week is {int(future_sales)} units. "
              f"Should restocking be done? Suggest a human-readable restocking advice.")

    response = llm(prompt, max_length=100)[0]['generated_text']

    return f"🔮 Predicted Sales Next Week: {int(future_sales)} units\n\n🛒 Advice:\n{response}"

iface = gr.Interface(
    fn=inventory_advisor,
    inputs=[
        gr.Number(label="Product ID"),
        gr.Number(label="Current Inventory"),
        gr.Number(label="Units Sold Yesterday")
    ],
    outputs="text",
    title="📦 Real-Time Inventory Management (Multi-Product)",
    description="Enter product ID, current stock, and yesterday's sales. Get AI-based restocking advice!"
)

if __name__ == "__main__":
    iface.launch()