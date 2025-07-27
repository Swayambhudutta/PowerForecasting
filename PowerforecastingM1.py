
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
st.title("🔮 Short-Term Intra-Day Forecast of Power Demand")

uploaded_file = st.file_uploader("📤 Upload Power Demand Excel File", type=["xlsx"])

if uploaded_file is not None:
    df = pd.read_excel(uploaded_file, engine='openpyxl')
    df['Datetime'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['Time'].astype(str))
    df.sort_values(by=['State', 'Datetime'], inplace=True)

    mw_rates = {
        'Maharashtra': 5.2, 'Tamil Nadu': 4.8, 'Karnataka': 5.0, 'Gujarat': 5.1,
        'West Bengal': 4.9, 'Rajasthan': 5.3, 'Uttar Pradesh': 4.7,
        'Kerala': 5.4, 'Punjab': 5.0, 'Bihar': 4.6
    }

    st.sidebar.header("⚙️ Model Configuration")
    selected_model = st.sidebar.selectbox("Choose Forecasting Model", [
        "SARIMAX", "RandomForest", "LinearRegression", "SVR", "XGBoost", "LSTM", "GRU", "Hybrid"
    ])
    st.sidebar.subheader("📊 Accuracy Metrics")

    state = st.selectbox("📍 Select State", df['State'].unique())
    rate = mw_rates.get(state, 5.0)

    state_df = df[df['State'] == state].sort_values('Datetime')
    series = state_df['Power Demand (MW)'].values[:100]
    dates = state_df['Datetime'].values[:100]
    train, test = series[:70], series[70:]

    def create_features(data, window=5):
        X, y = [], []
        for i in range(window, len(data)):
            X.append(data[i-window:i])
            y.append(data[i])
        return np.array(X), np.array(y)

    def train_model(model_name, X_train, y_train, X_test, scaler, train, test):
        if model_name == "SARIMAX":
            model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(0, 0, 0, 0))
            model_fit = model.fit(disp=False)
            forecast = model_fit.forecast(steps=30)
        elif model_name in ["RandomForest", "LinearRegression", "SVR", "XGBoost"]:
            if model_name == "RandomForest":
                model = RandomForestRegressor(n_estimators=100, random_state=42)
            elif model_name == "LinearRegression":
                model = LinearRegression()
            elif model_name == "SVR":
                model = SVR(kernel='rbf')
            elif model_name == "XGBoost":
                model = XGBRegressor(n_estimators=100, random_state=42)

            model.fit(X_train, y_train)
            forecast_scaled = model.predict(X_test)
            forecast = scaler.inverse_transform(forecast_scaled.reshape(-1, 1)).flatten()
            test = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
        elif model_name in ["LSTM", "GRU", "Hybrid"]:
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

            model = TimeSeriesModel(model_name)
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

        baseline = np.full_like(test, np.mean(train))
        mw_savings = np.sum(baseline - forecast)
        financial_gain = mw_savings * rate
        yearly_gain = financial_gain * 365
        return forecast, test, financial_gain, yearly_gain

    scaler = MinMaxScaler()
    scaled_series = scaler.fit_transform(series.reshape(-1, 1)).flatten()
    window = 5
    X_train, y_train = create_features(scaled_series[:70], window)
    X_test, y_test = create_features(scaled_series[70-window:100], window)

    forecast, test, financial_gain, yearly_gain = train_model(
        selected_model, X_train, y_train, X_test, scaler, train, test
    )

    rmse = np.sqrt(mean_squared_error(test, forecast))
    mae = mean_absolute_error(test, forecast)
    r2_raw = r2_score(test, forecast)
    r2 = max(0.0, r2_raw)

    st.sidebar.write(f"**R² Score**: {r2:.2f}")
    st.sidebar.write(f"**RMSE**: {rmse:.2f}")
    st.sidebar.write(f"**MAE**: {mae:.2f}")

    baseline = np.full_like(test, np.mean(train))
    mw_savings = np.sum(baseline - forecast)

    col1, col2 = st.columns([3, 1])

    with col1:
        st.subheader(f"📈 Forecast vs Actual using {selected_model}")
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
        ax.set_ylabel("Power Demand (MW)")
        plt.xticks(rotation=45)
        st.pyplot(fig)
        st.caption("📌 Disclaimer: Model trained on 70 blocks of 15-minute data and predicted for the last 30 blocks.")

    with col2:
        st.subheader("💰 Financial Highlights")
        st.markdown(f"<h5><strong>MW Savings:</strong> {mw_savings:.2f} MW</h5>", unsafe_allow_html=True)
        st.markdown(f"<h5><strong>Daily Financial Gain:</strong> ₹{financial_gain:,.2f}</h5>", unsafe_allow_html=True)
        st.markdown(f"<h5><strong>Estimated Yearly Gain:</strong> ₹{yearly_gain:,.2f}</h5>", unsafe_allow_html=True)
        st.caption(f"💡 Rate per MW in {state}: ₹{rate:.2f}")

    if st.button("Optimize"):
        best_model = None
        best_gain = -np.inf
        for model_name in ["SARIMAX", "RandomForest", "LinearRegression", "SVR", "XGBoost", "LSTM", "GRU", "Hybrid"]:
            try:
                forecast_opt, test_opt, gain_opt, _ = train_model(
                    model_name, X_train, y_train, X_test, scaler, train, test
                )
                if gain_opt > best_gain:
                    best_gain = gain_opt
                    best_model = model_name
            except Exception as e:
                continue
        if best_model:
            st.success(f"✅ Optimized Model: {best_model}")
            st.markdown(f"<h5><strong>Optimized Daily Financial Gain:</strong> ₹{best_gain:,.2f}</h5>", unsafe_allow_html=True)
        else:
            st.error("❌ Optimization failed. Please check your data or model configurations.")
else:
    st.info("Please upload a power demand Excel file to begin.")
