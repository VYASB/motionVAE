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


def train(model, train_loader, optimizer):
    model.train()
    train_loss = 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optimizer.zero_grad()
    
    recon_batch, mu, log_var, loss = model(train_loader)
    
    loss.backward()
    optimizer.step()
        
    total_loss += loss.item()
            
    #print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader)}")
    return train_loss


def test(model, test_loader):
    model.eval()
    test_loss= 0
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.cuda()
            recon, mu, log_var, final_loss = model(data)
            
            # sum up batch loss
            test_loss += final_loss.item()
    test_loss /= len(test_loader)
    print('====> Test set loss: {:.4f}'.format(test_loss))
    return test_loss