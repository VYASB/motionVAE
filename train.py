import torch
import torch.optim as optim
import torch.nn as nn
import model

import torchvision.transforms as transforms
import torchvision
import matplotlib

from utils import MotionDataParser, save_loss_plot
from engine import train, test

# Example usage
input_dim = 156  # Assuming each SMPL pose has 72 dimensions (e.g., 24 joints with 3 angles each)
hidden_dim = 128
latent_dim = 32

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
vae = model.VAE(input_dim=input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim).to(device)

lr = 0.001
epochs = 100
batch_size = 100
optimizer = optim.Adam(vae.parameters())

data_dir = 'D:/ShapeShifter23/motionVAE/data'
parser = MotionDataParser(data_dir)
motions = parser.load_data()
train_motions, test_motions = parser.split_data(motions, split_ratio=0.8)

train_loss = []
valid_loss =[]
print(f"The device is: {device}")

for epoch in range(epochs):
     print(f"Epoch {epoch+1} of {epochs}")
     train_epoch_loss = train(vae, train_motions, epoch, optimizer, device)
     valid_epoch_loss = test(vae, test_motions, device)

     train_loss.append(train_epoch_loss)
     valid_loss.append(valid_epoch_loss)

     ##Save the reconstructed image from the validation loop
    # save_reconstructed_images(recon_images, epoch+1)

     ##convert the reconstructed images to PyTorch image grid format
     #image_grid = make_grid(recon_images.detach().cpu())
   #  grid_images.append(image_grid)

     print(f"Train Loss: {train_epoch_loss: .4f}")
     print(f"Val Loss: {valid_epoch_loss: .4f}")

save_loss_plot(train_loss, valid_loss)
torch.save(vae.state_dict(), 'D:/ShapeShifter23/ShapeShifterTesting/outputs/model.pt')

print('TRAINING COMPLETE')