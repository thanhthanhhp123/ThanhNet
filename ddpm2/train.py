import torch
import os
import torch.nn as nn

from ddpm2.loss import get_loss

def trainer(model, loader, device = 'cuda' if torch.cuda.is_available() else 'cpu',
            epochs = 100, save_every = 10):
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001, weight_decay=1e-5)
    if not os.path.exists('models'):
        os.makedirs('models')
    model = model.to(device)

    for epoch in range(epochs):
        for obj in loader:
            image = obj['image'].to(device)
            optimizer.zero_grad()
            t = torch.randint(0, 200, (3,), device=device)
            loss = get_loss(model, image, t)
            loss.backward()
            optimizer.step()
        if epoch % save_every == 0:
            torch.save(model.state_dict(), f'models/model_{epoch}.pt')
        print(f'Epoch {epoch} Loss: {loss.item()}')

        
