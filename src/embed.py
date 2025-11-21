import torch
from PIL import Image
import mobileclip
import numpy as np

class MobileCLIP2Embedder:
    def __init__(self, ckpt_path="models/mobileclip_s2/mobileclip_s2.pt", device="cpu"):
        self.device = device

        # Load model + preprocessor
        self.model, _, self.preprocess = mobileclip.create_model_and_transforms(
            'mobileclip_s2',
            pretrained=ckpt_path
        )

        self.model = self.model.to(device).eval()

    def embed(self, img_bgr):
        # Convert OpenCV BGR -> RGB PIL image
        img = Image.fromarray(img_bgr[:, :, ::-1])

        # Preprocess (from MobileCLIP API)
        img_t = self.preprocess(img).unsqueeze(0).to(self.device)

        # Encode using MobileCLIP2
        with torch.no_grad():
            feats = self.model.encode_image(img_t)
            feats = feats / feats.norm(dim=-1, keepdim=True)

        # Return numpy vector
        return feats.cpu().numpy()[0]