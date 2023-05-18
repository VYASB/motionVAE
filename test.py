import torch
import torch.nn as nn
import scipy
import numpy

class MotionEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(MotionEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, latent_dim)
    
    def forward(self, motion):
        # motion shape: (batch_size, seq_len, input_dim)
        batch_size, seq_len, _ = motion.size()
        
        # Initialize hidden state and cell state
        hidden_state = torch.zeros(1, batch_size, self.hidden_dim)
        cell_state = torch.zeros(1, batch_size, self.hidden_dim)
        
        # Pass the motion sequence through the LSTM
        _, (final_hidden_state, _) = self.lstm(motion, (hidden_state, cell_state))
        
        # Extract the final hidden state
        final_hidden_state = final_hidden_state.squeeze(0)
        
        # Pass the final hidden state through the fully connected layer
        latent_representation = self.fc(final_hidden_state)
        
        return latent_representation

# Example usage
input_dim = 156  # Assuming each SMPL pose has 72 dimensions (e.g., 24 joints with 3 angles each)
hidden_dim = 128
latent_dim = 32

# Create an instance of the motion encoder
encoder = MotionEncoder(input_dim, hidden_dim, latent_dim)

# Generate random motion data for demonstration purposes
batch_size = 1  # For simplicity, considering a single motion sequence
seq_len = 20
motion_data = torch.randn(batch_size, seq_len, input_dim)

# Pass the motion data through the encoder
latent_representation = encoder(motion_data)

# Print the latent representation
print(latent_representation.shape)