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

def final_loss(recon_x, x, mu, log_var):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return BCE + KLD

def train(model, dataloader, epoch, optimizer):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(dataloader):
        data = data.cuda()
        optimizer.zero_grad()
        
        recon_batch, mu, log_var = model(data)
        loss = final_loss(recon_batch, data, mu, log_var)
        
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(dataloader.dataset),
                100. * batch_idx / len(dataloader), loss.item() / len(data)))
    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(dataloader.dataset)))
    return train_loss / len(dataloader.dataset)

def test(model, dataloader):
    model.eval()
    test_loss= 0
    with torch.no_grad():
        for data, _ in dataloader:
            data = data.cuda()
            recon, mu, log_var = model(data)
            
            # sum up batch loss
            test_loss += final_loss(recon, data, mu, log_var).item()
    test_loss /= len(dataloader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))
    return test_loss