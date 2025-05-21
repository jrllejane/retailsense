# heatmap_maker/utils/feature_extractor.py

import torch
from torchreid import models
from torchreid.utils import load_pretrained_weights

class FeatureExtractor:
    def __init__(self, model_name='osnet_x1_0', model_path='', device='cpu'):
        self.device = torch.device(device)
        self.model = models.build_model(
            name=model_name,
            num_classes=1000,  # Dummy
            loss='softmax',
            pretrained=False
        )
        if model_path:
            load_pretrained_weights(self.model, model_path)

        self.model.to(self.device)
        self.model.eval()

    def __call__(self, image_tensor):
        with torch.no_grad():
            return self.model(image_tensor.to(self.device))
