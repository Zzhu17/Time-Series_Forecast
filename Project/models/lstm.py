"""
LSTM_Model:
Reconstructed for time series trend forecasting.
Supports configurable input size, hidden units, dropout, and layer depth.
Optimized to return the last timestep output for prediction.
"""
import torch
import torch.nn as nn

class lstm_model(nn.Module):
    def __init__(self, input_size, hidden_size, dropout=0.0, num_layers=1):
        """
        input_size: Dimension of each input time step
        hidden_size: Number of features in LSTM hidden state
        dropout: Dropout probability (ignored if num_layers=1)
        num_layers: Number of stacked LSTM layers
        """
        super(lstm_model, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        """
        x: Input tensor of shape (batch_size, sequence_length, input_size)
        Returns:
            Tensor of shape (batch_size, 1) corresponding to the predicted value for the last timestep
        """
        assert x.ndim == 3, f"Expected input of shape (batch, seq_len, input_size), but got {x.shape}"
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # use the last timestep
        return out

    def __str__(self):
        return f"LSTM_Model(input_size={self.lstm.input_size}, hidden_size={self.lstm.hidden_size}, num_layers={self.lstm.num_layers})"

if __name__ == "__main__":
    # Step 5: Device check and summary
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = lstm_model(input_size=5, hidden_size=64, dropout=0.1, num_layers=2).to(device)
    dummy_input = torch.randn(4, 10, 5).to(device)
    with torch.no_grad():
        output = model(dummy_input)
    print(f"Output shape: {output.shape}")