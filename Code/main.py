import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from mamba_ssm import Mamba
from torch.nn import LSTM
import numpy as np
import pandas as pd
import torch.optim as optim
import matplotlib.pyplot as plt
from dataset import load_data, getBatteryCapacity, getBatteryValues
import os


# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
# Build the relative path to the dataset
dataset_path = os.path.join(script_dir, "../data/downloaded_files/")

print("Dataset loaded successfully!")
Battery = load_data(dataset_path)

print(Battery)

### MambaNet ###

class PositionalEncoding(nn.Module):
    def __init__(self, feature_len, feature_size, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(feature_len, feature_size)
        position = torch.arange(0, feature_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, feature_size, 2).float() * (-math.log(10000.0) / feature_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])


class MambaBlock(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super(MambaBlock, self).__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.mamba = Mamba(d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        # Mamba block + residual
        x = x + self.dropout1(self.mamba(self.norm1(x)))
        # Feedforward block + residual
        x = x + self.dropout2(self.ff(self.norm2(x)))
        return x


class MambaNet(nn.Module):
    def __init__(self, feature_size=1, seq_len=16, d_model=256, num_layers=2, dropout=0.1, use_pos_encoding=True):
        super(MambaNet, self).__init__()
        self.input_proj = nn.Linear(feature_size, d_model)

        self.use_pos_encoding = use_pos_encoding
        if use_pos_encoding:
            self.pos_encoding = PositionalEncoding(seq_len, d_model, dropout)

        self.encoder = nn.Sequential(*[
            MambaBlock(d_model=d_model, dropout=dropout)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(d_model)
        self.output_layer = nn.Linear(d_model, 1)

        self._init_weights()

    def forward(self, x):
        x = self.input_proj(x)  # (B, S, d_model)
        if self.use_pos_encoding:
            x = self.pos_encoding(x)

        x = self.encoder(x)
        x = self.norm(x)
        x = x.mean(dim=1)       # Global average pooling
        return self.output_layer(x)  # (B, 1)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    
### AutoReformer ###

# Series Decomposition (Trend + Seasonal)
class SeriesDecomposition(nn.Module):
    def __init__(self, kernel_size):
        super(SeriesDecomposition, self).__init__()
        self.moving_avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)

    def forward(self, x):
        trend = self.moving_avg(x.transpose(1, 2)).transpose(1, 2)
        seasonal = x - trend
        return seasonal, trend


# Auto-Correlation Layer (simplified)
class AutoCorrelationLayer(nn.Module):
    def __init__(self, d_model):
        super(AutoCorrelationLayer, self).__init__()
        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        Q = self.query_proj(x)
        K = self.key_proj(x)
        V = self.value_proj(x)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(Q.size(-1))
        attn = torch.softmax(scores, dim=-1)
        context = torch.matmul(attn, V)
        return self.output_proj(context)


# Autoformer Encoder Layer
class AutoformerEncoderLayer(nn.Module):
    def __init__(self, d_model, kernel_size=3):
        super(AutoformerEncoderLayer, self).__init__()
        self.decomp = SeriesDecomposition(kernel_size)
        self.auto_corr = AutoCorrelationLayer(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )

    def forward(self, x):
        seasonal, trend = self.decomp(x)
        seasonal = self.auto_corr(seasonal)
        seasonal = self.ff(seasonal)
        return seasonal + trend


# Full Autoformer Model
class AutoformerNet(nn.Module):
    def __init__(self, feature_size=1, seq_len=16, d_model=256, num_layers=2, kernel_size=3):
        super(AutoformerNet, self).__init__()
        self.input_proj = nn.Linear(feature_size, d_model)
        self.pos_encoding = PositionalEncoding(feature_len=seq_len, feature_size=d_model)
        self.encoder_layers = nn.ModuleList([
            AutoformerEncoderLayer(d_model=d_model, kernel_size=kernel_size)
            for _ in range(num_layers)
        ])
        self.output_layer = nn.Linear(d_model, 1)

    def forward(self, x):
        # x: (batch, seq_len, feature_size)
        x = self.input_proj(x)                      # -> (batch, seq_len, d_model)
        x = self.pos_encoding(x)                    # Add positional info

        for layer in self.encoder_layers:
            x = layer(x)                            # Pass through encoder layers

        out = x.mean(dim=1)                         # Global average pooling
        out = self.output_layer(out)                # Final prediction
        return out

### Dlinear ###

class AdvancedDLinear(nn.Module):
    def __init__(self, feature_size=1, seq_len=16, hidden_size=256, num_layers=4, dropout_rate=0.2, l2_lambda=1e-4):
        super(AdvancedDLinear, self).__init__()
        self.seq_len = seq_len
        self.feature_size = feature_size
        self.input_dim = seq_len * feature_size
        self.l2_lambda = l2_lambda  # L2 regularization factor

        # Trend Component
        trend_layers = []
        for _ in range(num_layers - 1):
            trend_layers.extend([
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
            ])

        self.trend_layer = nn.Sequential(
            nn.Linear(self.input_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            *trend_layers,
            nn.Linear(hidden_size, 1)
        )

        # Seasonal Component
        season_layers = []
        for _ in range(num_layers - 1):
            season_layers.extend([
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
            ])

        self.season_layer = nn.Sequential(
            nn.Linear(self.input_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            *season_layers,
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x):
        # x: (batch_size, seq_len, feature_size)
        batch_size = x.size(0)
        x = x.view(batch_size, -1)  # Flatten to (batch_size, seq_len * feature_size)

        trend = self.trend_layer(x)
        season = self.season_layer(x - trend)
        return trend + season
    
### XLSTM ###

class XLSTMBlock(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super(XLSTMBlock, self).__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.xlstm = nn.LSTM(input_size=d_model, hidden_size=d_model, num_layers=1, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        # XLSTM block + residual
        x_norm = self.norm1(x)
        x_lstm, _ = self.xlstm(x_norm)  # XLSTM expects inputs in (batch_size, seq_len, d_model)
        x = x + self.dropout1(x_lstm)

        # Feedforward block + residual
        x = x + self.dropout2(self.ff(self.norm2(x)))
        return x


class XLSTMNet(nn.Module):
    def __init__(self, feature_size=1, seq_len=16, d_model=256, num_layers=2, dropout=0.1, use_pos_encoding=True):
        super(XLSTMNet, self).__init__()
        self.input_proj = nn.Linear(feature_size, d_model)

        self.use_pos_encoding = use_pos_encoding
        if use_pos_encoding:
            self.pos_encoding = PositionalEncoding(seq_len, d_model, dropout)

        self.encoder = nn.Sequential(*[
            XLSTMBlock(d_model=d_model, dropout=dropout)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(d_model)
        self.output_layer = nn.Linear(d_model, 1)

        self._init_weights()

    def forward(self, x):
        x = self.input_proj(x)  # (B, S, d_model)
        if self.use_pos_encoding:
            x = self.pos_encoding(x)

        x = self.encoder(x)
        x = self.norm(x)
        x = x.mean(dim=1)       # Global average pooling
        return self.output_layer(x)  # (B, 1)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

# dataset_path = os.path.join(script_dir, "../data/downloaded_files/")

# --- Load the trained model ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_AutoReformer = AutoformerNet(feature_size=1, seq_len=16, d_model=256, num_layers=2, kernel_size=3)
model_AutoReformer.load_state_dict(torch.load(os.path.join(script_dir, "../data/downloaded_files/AutoReformer.pth"), map_location=device))
model_AutoReformer.to(device)

model_Mamba = MambaNet(feature_size=1, seq_len=16, d_model=256, num_layers=2, dropout=0.0)
model_Mamba.load_state_dict(torch.load(os.path.join(script_dir, "../data/downloaded_files/Mamba.pth"), map_location=device))
model_Mamba.to(device)

model_Dlinear = AdvancedDLinear(feature_size=1, seq_len=16, hidden_size=256, num_layers=4, dropout_rate=0.3, l2_lambda=1e-3)
model_Dlinear.load_state_dict(torch.load(os.path.join(script_dir, "../data/downloaded_files/Adv_Dlinear.pth"), map_location=device))
model_Dlinear.to(device)

model_xLSTM = XLSTMNet(feature_size=1, seq_len=16, d_model=256, num_layers=2, dropout=0.0)
model_xLSTM.load_state_dict(torch.load(os.path.join(script_dir, "../data/downloaded_files/XLSTM.pth"), map_location=device))
model_xLSTM.to(device)

def generate_autoregressive_output(model, Battery, target_battery_id='B0005', window_size=16, pred_steps=117, device='cpu'):
    """
    Generates autoregressive prediction output for a specific battery using the given model.

    Args:
        model (torch.nn.Module): Trained PyTorch model.
        Battery (dict): Dictionary containing battery data in format {battery_id: (cycles, capacities)}.
        target_battery_id (str): ID of the battery to predict.
        window_size (int): Size of the input window.
        pred_steps (int): Number of autoregressive steps to predict.
        device (str): 'cpu' or 'cuda'.

    Returns:
        np.ndarray: Array of predicted capacities.
    """
    model.eval()
    with torch.no_grad():
        capacities = Battery[target_battery_id][1]
        capacities_tensor = torch.tensor(capacities, dtype=torch.float32)

        # Initialize input sequence
        input_seq = capacities_tensor[:window_size].clone()
        sequence = input_seq.tolist()

        # Autoregressive prediction
        for _ in range(pred_steps):
            x_input = input_seq[-window_size:].unsqueeze(1).unsqueeze(0).to(device)  # shape: [1, window_size, 1]
            pred = model(x_input)
            pred_value = pred.item()
            sequence.append(pred_value)
            input_seq = torch.cat([input_seq, torch.tensor([pred_value])])

    return np.array(sequence)

# Printing for testing purposes

print(generate_autoregressive_output(model_Mamba, Battery, target_battery_id='B0005', window_size=16, pred_steps=len(Battery['B0005'][1])-16, device=device).shape)
print(generate_autoregressive_output(model_AutoReformer, Battery, target_battery_id='B0005', window_size=16, pred_steps=117, device=device))
print(generate_autoregressive_output(model_Dlinear, Battery, target_battery_id='B0005', window_size=16, pred_steps=117, device=device))
print(generate_autoregressive_output(model_xLSTM, Battery, target_battery_id='B0005', window_size=16, pred_steps=117, device=device).shape)

#Ensemble the predictions

# --- Define LSTM with Attention ---
class LSTMWithAttention(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=3, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout=dropout, batch_first=True)
        self.attn = nn.Linear(hidden_size, 1)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attn_weights = torch.softmax(self.attn(lstm_out), dim=1)
        context = torch.sum(attn_weights * lstm_out, dim=1)
        return self.fc(context).squeeze(-1)

# --- Define Log-Cosh Loss ---
class LogCoshLoss(nn.Module):
    def forward(self, pred, target):
        return torch.mean(torch.log(torch.cosh(pred - target + 1e-12)))

# --- Sliding Window Sequence Generator ---
def create_sequences(X, y, window_size):
    X_seq, y_seq = [], []
    for i in range(len(X) - window_size):
        X_seq.append(X[i:i + window_size])
        y_seq.append(y[i + window_size])
    return np.array(X_seq), np.array(y_seq)

# --- Train Function ---
def train_lstm_ensemble(X_all, y_all, epochs=1000, lr=0.001):
    X = np.concatenate(X_all, axis=0)
    y = np.concatenate(y_all, axis=0)

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    input_size = X.shape[-1]
    model = LSTMWithAttention(input_size=input_size)
    criterion = LogCoshLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(X_tensor)
        loss = criterion(output, y_tensor)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} - Loss: {loss.item():.4f}")

    return model

# --- Forecast Function ---
def lstm_ensemble_forecast(model, model_outputs, window_size):
    X = np.stack(model_outputs, axis=-1)
    time_index = np.linspace(0, 1, len(model_outputs[0]))
    X = np.concatenate([X, time_index[:, None]], axis=-1)

    X_seq, _ = create_sequences(X, np.zeros(len(X)), window_size)
    X_tensor = torch.tensor(X_seq, dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        preds = model(X_tensor).numpy()

    return np.concatenate([np.zeros(window_size), preds])

# --- Main ---
battery_ids = ['B0005', 'B0006', 'B0007', 'B0018']
X_all, y_all = [], []
window_size = 16

for battery_id in battery_ids:
    predictions_AutoReformer = generate_autoregressive_output(model_AutoReformer, Battery, battery_id, window_size, len(Battery[battery_id][1]) - window_size, device=device)
    predictions_Mamba = generate_autoregressive_output(model_Mamba, Battery, battery_id, window_size, len(Battery[battery_id][1]) - window_size, device=device)
    predictions_Dlinear = generate_autoregressive_output(model_Dlinear, Battery, battery_id, window_size, len(Battery[battery_id][1]) - window_size, device=device)
    predictions_xLSTM = generate_autoregressive_output(model_xLSTM, Battery, battery_id, window_size, len(Battery[battery_id][1]) - window_size, device=device)

    model_outputs = [
        predictions_AutoReformer,
        predictions_Mamba,
        predictions_Dlinear,
        predictions_xLSTM
    ]

    time_index = np.linspace(0, 1, len(model_outputs[0]))
    X = np.stack(model_outputs + [time_index], axis=-1)
    y = np.array(Battery[battery_id][1][:len(model_outputs[0])])

    X_seq, y_seq = create_sequences(X, y, window_size)
    X_all.append(X_seq)
    y_all.append(y_seq)

# --- Train Model ---
shared_model = train_lstm_ensemble(X_all, y_all)

# --- Evaluate ---
fig, axs = plt.subplots(2, 2, figsize=(14, 10))
axs = axs.flatten()

for i, battery_id in enumerate(battery_ids):
    predictions_AutoReformer = generate_autoregressive_output(model_AutoReformer, Battery, battery_id, window_size, len(Battery[battery_id][1]) - window_size, device=device)
    predictions_Mamba = generate_autoregressive_output(model_Mamba, Battery, battery_id, window_size, len(Battery[battery_id][1]) - window_size, device=device)
    predictions_Dlinear = generate_autoregressive_output(model_Dlinear, Battery, battery_id, window_size, len(Battery[battery_id][1]) - window_size, device=device)
    predictions_xLSTM = generate_autoregressive_output(model_xLSTM, Battery, battery_id, window_size, len(Battery[battery_id][1]) - window_size, device=device)

    model_outputs = [
        predictions_AutoReformer,
        predictions_Mamba,
        predictions_Dlinear,
        predictions_xLSTM
    ]

    target_values = np.array(Battery[battery_id][1][:len(model_outputs[0])])
    final_predictions = lstm_ensemble_forecast(shared_model, model_outputs, window_size)

    axs[i].plot(target_values[17:], label='Original Sequence')
    axs[i].plot(final_predictions[17:], label='LSTM+Attention Prediction')
    axs[i].set_title(f'Battery {battery_id}')
    axs[i].set_xlabel('Time Steps')
    axs[i].set_ylabel('Value')
    axs[i].legend()
    axs[i].grid(True)

plt.tight_layout()
plt.show()