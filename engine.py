from tqdm import tqdm
import torch.nn.functional as F
import torch


#     """

#     This fxn will add the reconstructuion loss (BCELoss) and the KL_Divergence
#     Kl-Divergence  = 0.5* sum(1+log(sigma^2) - mu^2 - sigma^2)

#     :bce_loss = reconstruction loss
#     :mu = the mena from the latent vector
#     :logvar = log variance from the latent vector (standard deviation)
#     """

#     BCE = bce_loss
#     KLD = -0.5*torch.sum(1+logvar - mu.pow(2) - logvar.exp())

#     return BCE+KLD


def train(model, train_data, optimizer, device):
    model.train()
    train_loss = 0.0
    
    # Move the training data to the specified device
    train_data = train_data.to(device)
    
    optimizer.zero_grad()
    
    # Forward pass
    reconstructed_motion, mu, var, loss = model(train_data)
    
    # Backward pass
    loss.backward()
    optimizer.step()
    
    train_loss += loss.item()
    
    return train_loss

def test(model, test_data, device):
    model.eval()
    test_loss = 0.0
    
    # Move the test data to the specified device
    test_data = test_data.to(device)
    
    # Forward pass
    with torch.no_grad():
        reconstructed_motion, mu, var, loss = model(test_data)
    
    test_loss += loss.item()
    
    return test_loss