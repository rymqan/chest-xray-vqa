import torch
import torch.nn as nn


class FusionModule(nn.Module):
    def __init__(self, image_dim, text_dim, fused_dim):
        """
        Fusion module to combine image and text features.

        Args:
            image_dim (int): Dimensionality of image features.
            text_dim (int): Dimensionality of text features.
            fused_dim (int): Dimensionality of the fused features.
        """
        super(FusionModule, self).__init__()
        self.fc = nn.Linear(image_dim + text_dim, fused_dim)
        self.activation = nn.ReLU()

    def forward(self, image_features, text_features):
        """
        Forward pass for the fusion module.

        Args:
            image_features (torch.Tensor): Batch of image features (shape: [batch_size, image_dim]).
            text_features (torch.Tensor): Batch of text features (shape: [batch_size, text_dim]).

        Returns:
            torch.Tensor: Fused features (shape: [batch_size, fused_dim]).
        """
        # Concatenate image and text features
        combined_features = torch.cat((image_features, text_features), dim=1)
        # Apply fully connected layer and activation
        fused_features = self.fc(combined_features)
        fused_features = self.activation(fused_features)
        return fused_features
