
import torch
import torch.nn as nn
from torchvision import models as torch_models

class ROIBasedDenseNetModel(nn.Module):
    def __init__(self, num_classes, num_channels):
        super(ROIBasedDenseNetModel, self).__init__()

        # Pretrained DenseNet model
        self.roi_model = torch_models.densenet201(weights='DEFAULT')
        self.roi_model.features.conv0 = nn.Conv2d(num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)  # Set the number of input channels to 2
        self.roi_model.features.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # Downsample output to a fixed size
        num_features_roi = self.roi_model.classifier.in_features
        self.roi_model.classifier = nn.Identity()  # Remove the final layer

        # Fully connected layers
        self.fc1 = nn.Linear(num_features_roi, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, roi):
        # Feature extraction from the ROI
        roi_features = self.roi_model.features(roi)
        roi_features = torch.flatten(roi_features, 1)  # Flattening

        # Classification through fully connected layers
        x = torch.relu(self.fc1(roi_features))
        x = self.fc2(x)
        return x
