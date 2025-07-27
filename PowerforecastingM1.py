import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX
from xgboost import XGBRegressor
import torch
import torch.nn as nn

st.set_page_config(layout="wide")
st.title("🔮 Power Demand Forecasting & Financial Insights")

# Load data
df = pd.read_excel("sample_power_demand_data.xlsx", engine='openpyxl')
df['Datetime'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['Time'].astype(str))
df.sort_values(by=['State', 'Datetime'], inplace=True)

# Model mapping and MW rates
model_mapping = {
    'Maharashtra': 'GRU',
    'Tamil Nadu': 'LSTM',
    'Karnataka': 'Hybrid',
    'Gujarat': 'LSTM',
    'West Bengal': 'GRU',
    'Rajasthan': 'Hybrid',
    'Uttar Pradesh': 'GRU',
    'Kerala': 'Hybrid',
    'Punjab': 'LSTM',
    'Bihar': 'Hybrid'
}

mw_rates = {
    'Maharashtra': 5.2,
    'Tamil Nadu': 4.8,
    'Karnataka': 5.0,
    'Gujarat': 5.1,
    'West Bengal': 4.9,
    'Rajasthan': 5.3,
    'Uttar Pradesh': 4.7,
    'Kerala': 5.4,
    'Punjab': 5.0,
    'Bihar': 4.6
}

# Sidebar
state = st.sidebar.selectbox("📍 Select State", df['State'].unique())
model_type = model_mapping[state]
rate = mw_rates[state]

# Data preparation
state_df = df[df['State'] == state].sort_values('Datetime')
series = state_df['Power Demand (MW)'].values[:100]
dates = state_df['Datetime'].values[:100]
train, test = series[:70], series[70:]

# Feature creation
def create_features(data, window=5):
    X, y = [], []
    for i in range(window, len(data)):
        X.append(data[i-window:i])
        y.append(data[i])
    return np.array(X), np.array(y)

def create_dl_features(data, window=5):
    return create_features(data, window)

# Scaling
scaler = MinMaxScaler()
scaled_series = scaler.fit_transform(series.reshape(-1, 1)).flatten()
window = 5
X_train, y_train = create_features(scaled_series[:70], window)
X_test, y_test = create_features(scaled_series[70-window:100], window)

# Model training
if model_type in ["RandomForest", "LinearRegression", "SVR", "XGBoost"]:
    if model_type == "RandomForest":
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    elif model_type == "LinearRegression":
        model = LinearRegression()
    elif model_type == "SVR":
        model = SVR(kernel='rbf')
    elif model_type == "XGBoost":
        model = XGBRegressor(n_estimators=100, random_state=42)

    model.fit(X_train, y_train)
    forecast_scaled = model.predict(X_test)
    forecast = scaler.inverse_transform(forecast_scaled.reshape(-1, 1)).flatten()
    test = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

elif model_type in ["LSTM", "GRU", "Hybrid"]:
    X_train_torch = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1)
    y_train_torch = torch.tensor(y_train, dtype=torch.float32).unsqueeze(-1)
    X_test_torch = torch.tensor(X_test, dtype=torch.float32).unsqueeze(-1)

    class TimeSeriesModel(nn.Module):
        def __init__(self, model_type):
            super().__init__()
            if model_type == "LSTM":
                self.rnn = nn.LSTM(input_size=1, hidden_size=50, batch_first=True)
            elif model_type == "GRU":
                self.rnn = nn.GRU(input_size=1, hidden_size=50, batch_first=True)
            elif model_type == "Hybrid":
                self.rnn = nn.LSTM(input_size=1, hidden_size=50, batch_first=True)
                self.fc1 = nn.Linear(50, 25)
                self.fc2 = nn.Linear(25, 1)
            else:
                raise ValueError("Invalid model type")

            if model_type != "Hybrid":
                self.fc = nn.Linear(50, 1)

            self.model_type = model_type

        def forward(self, x):
            out, _ = self.rnn(x)
            out = out[:, -1, :]
            if self.model_type == "Hybrid":
                out = torch.relu(self.fc1(out))
                out = self.fc2(out)
            else:
                out = self.fc(out)
            return out

    model = TimeSeriesModel(model_type)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(50):
        model.train()
        optimizer.zero_grad()
        output = model(X_train_torch)
        loss = criterion(output, y_train_torch)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        forecast_scaled = model(X_test_torch).squeeze().numpy()

    forecast = scaler.inverse_transform(forecast_scaled.reshape(-1, 1)).flatten()
    test = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

# Baseline prediction
baseline = np.full_like(test, np.mean(train))

# Savings calculation
mw_savings = np.sum(baseline - forecast)
financial_gain = mw_savings * rate

# Layout
col1, col2 = st.columns([3, 1])

with col1:
    st.subheader(f"📈 Forecast vs Actual using {model_type}")
    plot_df = pd.DataFrame({
        'Datetime': dates[100 - len(test):100],
        'Actual': test,
        'Baseline': baseline,
        'Predicted': forecast
    })

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(data=plot_df, x='Datetime', y='Actual', label='Actual', ax=ax)
    sns.lineplot(data=plot_df, x='Datetime', y='Baseline', label='Baseline', ax=ax)
    sns.lineplot(data=plot_df, x='Datetime', y='Predicted', label='Predicted', ax=ax)
    plt.ylabel("Power Demand (MW)")
    plt.xticks(rotation=45)
    st.pyplot(fig)

with col2:
    st.subheader("💰 Financial Benefits")
    st.metric(label="MW Savings", value=f"{mw_savings:.2f} MW")
    st.metric(label="Estimated Financial Gain", value=f"₹{financial_gain:,.2f}")
    st.caption(f"💡 Rate per MW in {state}: ₹{rate:.2f}")
