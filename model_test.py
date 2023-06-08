import torch
import torch.nn as nn
class ClothEncoder(nn.Module):
    def __init__(self):
        super(ClothEncoder, self).__init__()
        self.cloth_encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 4, 4, 2, 1),
            nn.ReLU(),
            # nn.Conv2d(16, 4, 4, 2, 1),
            # nn.ReLU(),
        )
    def forward(self, x):
        return self.cloth_encoder(x)
    
cloth_encoder = ClothEncoder()
output = cloth_encoder(torch.randn(1,3,1024,768))
print(output.shape)