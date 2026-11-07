import torch
import torch.nn as nn

class TimeManagementUnit(nn.Module):
    def __init__(self, embed_dim=64):
        super().__init__()
        self.time_encoder = nn.Linear(4, embed_dim)  # hour, minute, day, month
        self.activation = nn.ReLU()
        
    def forward(self, x, time_info):
        # time_info: [batch_size, 4] - normalized time features
        time_embed = self.activation(self.time_encoder(time_info))
        # Combine with main input
        output = x + time_embed.unsqueeze(1)
        return output