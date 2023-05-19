import torch
import torch.nn as nn
import torch.optim as optim
from utils import DataParser
from engine import train, test
from tqdm import trange

class MotionAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(MotionAutoencoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        
        # Encoder layers
        self.encoder_lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.encoder_fc = nn.Linear(hidden_dim, latent_dim)
        self.encoder_fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.encoder_fc_var = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder layers
        self.decoder_fc = nn.Linear(latent_dim, hidden_dim)
        self.decoder_lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.decoder_output_layer = nn.Linear(hidden_dim, input_dim)
    
    def encode(self, motion):
        # motion shape: (batch_size, seq_len, input_dim)
        batch_size, seq_len, _ = motion.size()
        
        # Initialize hidden state and cell state for encoder
        hidden_state = torch.zeros(1, batch_size, self.hidden_dim)
        cell_state = torch.zeros(1, batch_size, self.hidden_dim)
        
        # Pass the motion sequence through the LSTM
        _, (final_hidden_state, _) = self.encoder_lstm(motion, (hidden_state, cell_state))
        
        # Extract the final hidden state
        final_hidden_state = final_hidden_state.squeeze(0)
        
        # Pass the final hidden state through the fully connected layers
        mu = self.encoder_fc_mu(final_hidden_state)
        var = self.encoder_fc_var(final_hidden_state)
        
        return mu, var
    
    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mean + eps * std
        return z
    
    def decode(self, latent_representation, seq_len):
        # latent_representation shape: (batch_size, latent_dim)
        batch_size = latent_representation.size(0)
        
        # Pass the latent representation through the fully connected layer
        hidden_state = self.decoder_fc(latent_representation)
        hidden_state = hidden_state.unsqueeze(1)
        
        # Initialize cell state for decoder
        cell_state = torch.zeros(1, batch_size, self.hidden_dim)
        
        # Initialize output sequence
        output_sequence = []
        
        # Generate motion sequence using LSTM
        for _ in range(seq_len):
            output, (hidden_state, cell_state) = self.decoder_lstm(hidden_state, (hidden_state, cell_state))
            output_sequence.append(output)
        
        # Convert output sequence to tensor
        output_sequence = torch.cat(output_sequence, dim=1)
        
        # Pass the output sequence through the output layer
        motion_sequence = self.decoder_output_layer(output_sequence)
        
        return motion_sequence
    
    def forward(self, motion):
        # motion shape: (batch_size, seq_len, input_dim)
        batch_size, seq_len, _ = motion.size()
        
        # Encode the motion sequence
        mu, var = self.encode(motion)
        
        # Reparameterization
        latent_representation = self.reparameterize(mu, var)
        
        # Decode the latent representation
        reconstructed_motion = self.decode(latent_representation, seq_len)
        
        # Reconstruction loss
        reconstruction_loss = nn.MSELoss()(reconstructed_motion, motion)
        
        # KL divergence loss
        kl_loss = -0.5 * torch.sum(1 + var - mu.pow(2) - var.exp())
        
        # Total loss
        total_loss = reconstruction_loss + kl_loss
        
        return reconstructed_motion, mu, var, total_loss


input_dim = 156  # Assuming each SMPL pose has 72 dimensions (e.g., 24 joints with 3 angles each)
hidden_dim = 128
latent_dim = 32

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
vae = MotionAutoencoder(input_dim=input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim).to(device)

lr = 0.001
epochs = 100
optimizer = optim.Adam(vae.parameters())

file_paths = 'D:/ShapeShifter23/motionVAE/data'
parser = DataParser(file_paths)
motions = DataParser(file_paths).get_motion()
#motion_data = torch.unsqueeze(torch.from_numpy(motion.get_motion()), dim=0)

train_motions, test_motions = parser.split_data(motions, split_ratio=0.8)

train_motion_data = torch.zeros(len(train_motions), train_motions.shape[1], train_motions.shape[2])
test_motion_data = torch.zeros(len(test_motions), test_motions.shape[1], test_motions.shape[2])

for i, motion_array in enumerate(train_motions):
    train_motion_data[i] = torch.from_numpy(motion_array)
for i, motion_array in enumerate(test_motions):
    test_motion_data[i] = torch.from_numpy(motion_array)

train_motion_data = train_motion_data.to(torch.float32)
test_motion_data = test_motion_data.to(torch.float32)

train_loss = []
valid_loss =[]
print(f"The device is: {device}")

for epoch in trange(epochs):
    #  print(f"Epoch {epoch+1} of {epochs}")
     train_epoch_loss = train(vae, train_motion_data, optimizer)
     valid_epoch_loss = test(vae, test_motion_data)

     train_loss.append(train_epoch_loss)
     valid_loss.append(valid_epoch_loss)

     ##Save the reconstructed image from the validation loop
    # save_reconstructed_images(recon_images, epoch+1)

     ##convert the reconstructed images to PyTorch image grid format
     #image_grid = make_grid(recon_images.detach().cpu())
   #  grid_images.append(image_grid)

     print(f"Train Loss: {train_epoch_loss: .4f}")
     print(f"Val Loss: {valid_epoch_loss: .4f}")

torch.save(vae.state_dict(), 'D:/ShapeShifter23/motionVAE/outputs/model.pt')

print('TRAINING COMPLETE')