import torch
import torch.nn as nn
import scipy
import numpy
from utils import DataParser

class MotionEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(MotionEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, latent_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_var = nn.Linear(hidden_dim, latent_dim)
    
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
        #latent_representation = self.fc(final_hidden_state)
        # We can return & check this 'latent_representation' down below

        # Pass the final hidden state through the fully connected layers
        mu = self.fc_mu(final_hidden_state)
        var = self.fc_var(final_hidden_state)
        
        return mu, var

##Some reparametrization
def reparameterize(self, mean, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    z = mean + eps * std
    return z

## Total Loss
def loss_function(self, reconstructed_motion, motion, mean, logvar):
        # Reconstruction loss
    reconstruction_loss = nn.MSELoss()(reconstructed_motion, motion)
        
        # KL divergence loss
    kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
        
        # Total loss
    total_loss = reconstruction_loss + kl_loss
        
    return total_loss


###Some Decoding

class MotionDecoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(MotionDecoder, self).__init__()
        self.hidden_dim = hidden_dim
        
        self.fc = nn.Linear(latent_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, latent_representation):
        # latent_representation shape: (batch_size, latent_dim)
        batch_size = latent_representation.size(0)
        
        # Pass the latent representation through the fully connected layer
        hidden_state = self.fc(latent_representation)
        hidden_state = hidden_state.unsqueeze(1)
        
        # Initialize cell state
        cell_state = torch.zeros(1, batch_size, self.hidden_dim)
        
        # Initialize output sequence
        output_sequence = []
        
        # Generate motion sequence using LSTM
        for _ in range(seq_len):
            output, (hidden_state, cell_state) = self.lstm(hidden_state, (hidden_state, cell_state))
            output_sequence.append(output)
        
        # Convert output sequence to tensor
        output_sequence = torch.cat(output_sequence, dim=1)
        
        # Pass the output sequence through the output layer
        motion_sequence = self.output_layer(output_sequence)
        
        return motion_sequence



# Example usage
input_dim = 156  # Assuming each SMPL pose has 72 dimensions (e.g., 24 joints with 3 angles each)
hidden_dim = 128
latent_dim = 32

# Create an instance of the motion encoder
encoder = MotionEncoder(input_dim, hidden_dim, latent_dim)

# Generate random motion data for demonstration purposes
batch_size = 30  # For simplicity, considering a single motion sequence
seq_len = 20
motion_data1 = torch.randn(batch_size, seq_len, input_dim)
file_paths = 'D:/ShapeShifter23/motionVAE/data'
motions = DataParser(file_paths).get_motion()

#motion_data = torch.unsqueeze(torch.from_numpy(motion.get_motion()), dim=0)
motion_data = torch.zeros(len(motions), motions.shape[1], motions.shape[2])

for i, motion_array in enumerate(motions):
    motion_data[i] = torch.from_numpy(motion_array)

motion_data = motion_data.to(torch.float32)

# Pass the motion data through the encoder
latent_representation = encoder(motion_data)


print(latent_representation.shape)