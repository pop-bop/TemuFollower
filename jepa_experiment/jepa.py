import torch
import torch.nn as nn

class JEPAEncoder(nn.Module):
    """
    Encodes the raw image into a compressed latent embedding.
    In a true JEPA, this would be trained via self-supervised joint-embedding prediction.
    """
    def __init__(self, embed_dim=64):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Linear(32, embed_dim)
        
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.fc(x)

class JEPADecoder(nn.Module):
    """
    Decodes the abstract embedding into actionable physical state values for the PID.
    Outputs: [position (-1.0 to 1.0), angle (-pi/2 to pi/2)]
    """
    def __init__(self, embed_dim=64):
        super().__init__()
        self.regressor = nn.Sequential(
            nn.Linear(embed_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )
        
    def forward(self, embedding):
        return self.regressor(embedding)

class LineJEPA(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = JEPAEncoder(embed_dim=64)
        self.decoder = JEPADecoder(embed_dim=64)
        
    def forward(self, x):
        """
        Takes in a normalized image tensor (B, 3, H, W).
        Returns the latent embedding and the decoded (position, angle).
        """
        emb = self.encoder(x)
        state = self.decoder(emb)
        return emb, state
