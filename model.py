"""
CNN Model - ResNet50 for 9-class anomaly classification
"""
import torch
import torch.nn as nn
from torchvision import models


class AnomalyClassifierCNN(nn.Module):
    """CNN classifier for time series anomaly detection from plots"""

    def __init__(self, num_classes=9, pretrained=True, backbone='resnet50'):
        super(AnomalyClassifierCNN, self).__init__()

        # Load pretrained backbone
        if backbone == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            feature_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()  # Remove final FC layer

        elif backbone == 'resnet34':
            self.backbone = models.resnet34(pretrained=pretrained)
            feature_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()

        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # Custom classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # Extract features
        features = self.backbone(x)

        # Classify
        output = self.classifier(features)

        return output


def create_model(num_classes=9, pretrained=True, backbone='resnet50', device='cpu'):
    """
    Create and initialize model

    Args:
        num_classes: Number of output classes
        pretrained: Use pretrained weights
        backbone: Backbone architecture
        device: Device to use

    Returns:
        model: Initialized model on device
    """
    model = AnomalyClassifierCNN(
        num_classes=num_classes,
        pretrained=pretrained,
        backbone=backbone
    )

    model = model.to(device)

    print("\n" + "=" * 70)
    print("  MODEL ARCHITECTURE")
    print("=" * 70)
    print(f"  Backbone: {backbone}")
    print(f"  Pretrained: {pretrained}")
    print(f"  Output classes: {num_classes}")
    print(f"  Device: {device}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print("=" * 70)

    return model


if __name__ == "__main__":
    import config
    model = create_model(
        num_classes=len(config.CLASSES),
        pretrained=True,
        backbone=config.BACKBONE,
        device=config.DEVICE
    )

    # Test forward pass
    dummy_input = torch.randn(2, 3, 224, 224).to(config.DEVICE)
    output = model(dummy_input)
    print(f"\nTest output shape: {output.shape}")
