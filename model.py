import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        
        # Encoder layers
        self.fc_encoder = nn.Linear(input_dim, hidden_dim)
        self.fc_mean = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder layers
        self.fc_decoder = nn.Linear(latent_dim, hidden_dim)
        self.fc_output = nn.Linear(hidden_dim, input_dim)
    
    def encode(self, motion):
        hidden_state = F.relu(self.fc_encoder(motion))
        mean = self.fc_mean(hidden_state)
        logvar = self.fc_logvar(hidden_state)
        return mean, logvar
    
    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mean + eps * std
        return z
    
    def decode(self, z):
        hidden_state = F.relu(self.fc_decoder(z))
        reconstructed_motion = self.fc_output(hidden_state)
        return reconstructed_motion
    
    def forward(self, motion):
        mean, logvar = self.encode(motion)
        z = self.reparameterize(mean, logvar)
        reconstructed_motion = self.decode(z)
        return reconstructed_motion, mean, logvar
    
    def loss_function(self, reconstructed_motion, motion, mean, logvar):
        # Reconstruction loss
        reconstruction_loss = nn.MSELoss()(reconstructed_motion, motion)
        
        # KL divergence loss
        kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
        
        # Total loss
        total_loss = reconstruction_loss + kl_loss
        
        return total_loss
