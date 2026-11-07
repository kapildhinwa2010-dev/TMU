import torch
import torch.nn as nn
import time

class TimeManagementUnit(nn.Module):
    def __init__(self, input_dim, hidden_dim=32):
        super().__init__()
        self.time_encoder = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        
    def get_time_features(self):
        now = time.time()
        hour = (now % 86400) / 86400
        minute = ((now % 3600) / 3600)
        second = ((now % 60) / 60)
        return torch.tensor([hour, minute, second]).unsqueeze(0)
    
    def forward(self, x):
        time_feat = self.get_time_features().to(x.device)
        time_embed = self.time_encoder(time_feat)
        return x + time_embed

# Integration with FluidLayer
class FluidLayerWithTMU(nn.Module):
    def __init__(self, original_model, embed_dim):
        super().__init__()
        self.model = original_model
        self.tmu = TimeManagementUnit(embed_dim)
        
    def forward(self, x):
        x = self.tmu(x)
        return self.model(x)