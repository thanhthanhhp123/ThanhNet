import torch.nn as nn
import torch
import torch.nn.functional as F
import os
import tqdm
import matplotlib.pyplot as plt

import utils
import net
import students
import common
from ddpm import DDPM
from ddpm2 import UNet, loss
import metrics

class ThanhNet(nn.Module):
    def __init__(self, device = 'cuda' if torch.cuda.is_available() else 'cpu',
                 train_backbone = False, epochs = 50):
        super(ThanhNet, self).__init__()
        self.pdn = students.get_pdn_small(padding = True)
        self.ddpm = DDPM()
        self.pdn.load_state_dict(torch.load('d_models/pdn.pth')['model'])
        self.pdn.to(device)
        self.ddpm.to(device)

        self.train_backbone = train_backbone
        self.device = device
        self.epochs = epochs

        self.ddpm_opt = torch.optim.Adam(self.ddpm.parameters(), lr = 1e-4, weight_decay=1e-5)
        self.ddpm_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.ddpm_opt, milestones=[80, 120], gamma=0.1)

        if train_backbone:
            self.pdn_opt = torch.optim.Adam(self.pdn.parameters(), lr = 1e-4, weight_decay=1e-5)
    
    def set_model_dir(self, model_dir):
        self.model_dir = model_dir
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

    def _train_ddpm(self, x):
        # self.pdn.train()
        # self.ddpm.train()
        # self.ddpm_opt.zero_grad()
        # t = torch.randint(0, 200, (3,), device=self.device)
        # loss = loss.get_loss(self.ddpm, x, t)
        # loss.backward()
        # self.ddpm_opt.step()

        x_ts = self.ddpm.generate_ts(len(x))
        x_a, x_b = self.ddpm.forward_noise(x, x_ts)

        x_ts = torch.from_numpy(x_ts).view(-1, 1).float().to(self.device)
        x_a = torch.Tensor(x_a).to(self.device)
        x_b = torch.Tensor(x_b).to(self.device)
        
        y_p = self.ddpm(x_a, x_ts)
        loss = torch.mean(torch.abs(y_p - x_b))
        self.ddpm_opt.zero_grad()
        loss.backward()
        self.ddpm_opt.step()
        self.ddpm_scheduler.step()
        return loss.item()

    def _inference(self, images):
        self.pdn.eval()
        self.ddpm.load_state_dict(torch.load(os.path.join(self.model_dir, 'ddpm.pth'))['model'])
        self.ddpm.eval()
        with torch.no_grad():
            images = images.to(self.device)
            features_map = self.pdn(images)
            
            x_ts = self.ddpm.generate_ts(len(features_map))
            x_a, x_b = self.ddpm.forward_noise(features_map, x_ts)

            x_ts = torch.from_numpy(x_ts).view(-1, 1).float().to(self.device)
            x_a = torch.Tensor(x_a).to(self.device)
            x_b = torch.Tensor(x_b).to(self.device)
            reconstructed_feature_map = self.ddpm(x_a, x_ts)

            anomaly_map = self.calculate_anomaly_map(features_map, reconstructed_feature_map)

            anomaly_map_resized = F.interpolate(anomaly_map, size = (224, 224), mode = 'bilinear', align_corners=False)

            anomaly_map_resized =(anomaly_map_resized - anomaly_map_resized.min()) / (anomaly_map_resized.max() - anomaly_map_resized.min())

            return anomaly_map_resized
    
    def calculate_anomaly_map(self, orginal_features, reconstructed_features, threshold = 0.5):
        anomaly_map = torch.mean((orginal_features - reconstructed_features) ** 2, dim=1, keepdim=True)

        return anomaly_map


    
    def train(self, training_loader):
        torch.cuda.empty_cache()
        for iter in tqdm.tqdm(range(self.epochs)):
            for obj in training_loader:
                obj['image'] = obj['image'].to(self.device)
                
                with torch.no_grad():
                    features_map = self.pdn(obj['image'])
                loss = self._train_ddpm(features_map)
            print(f'Epoch: {iter}, Loss: {loss}')



        torch.save(
            {'epoch': iter,
            'model': self.ddpm.state_dict(),
            'optimizer': self.ddpm_opt.state_dict(),
            'loss': loss},
            os.path.join(self.model_dir, 'ddpm.pth')
        )

    def forward(self, images):
        return self._inference(images)

        
if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Device: ', device)
    sub_ds  = [
        "bottle"
    ]
    ds = utils.dataset(
        'mvtec',
        'Dataset', subdatasets= sub_ds,
        batch_size=32
    )
    get_dataloaders = ds[1]
    list_of_dataloaders = get_dataloaders()
    bottle_loaders = list_of_dataloaders[0]

    train_loader = bottle_loaders['train']
    test_loader = bottle_loaders['test']
    model = ThanhNet(device=device)
    model.to(device)
    model.set_model_dir('d_models')
    # model.train(train_loader)

    anomaly_map = model(test_loader.dataset[50]['image'].unsqueeze(0))
    print(anomaly_map   )


    plt.figure(figsize=(10, 10))
    plt.subplot(1, 3, 1)
    plt.imshow(test_loader.dataset[50]['image'].permute(1, 2, 0).squeeze().cpu().numpy())
    plt.title('Original Image')
    plt.axis('off')

    print(test_loader.dataset[50]['anomaly'])

    plt.subplot(1, 3, 2)
    plt.imshow(test_loader.dataset[50]['mask'].permute(1, 2, 0).squeeze().cpu().numpy())
    plt.title('Mask')
    plt.axis('off')


    plt.subplot(1, 3, 3)
    plt.imshow(anomaly_map[0].permute(1, 2, 0).squeeze().cpu().numpy(), cmap='hot')
    plt.title('Anomaly Map')
    plt.axis('off')

    plt.show()
    



