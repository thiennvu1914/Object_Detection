"""
MobileCLIP Embedder Module
===========================
Generate embeddings for food classification.
"""
from pathlib import Path
import torch
import mobileclip
import numpy as np
from PIL import Image


class MobileCLIPEmbedder:
    """MobileCLIP-based embedding generator"""
    
    def __init__(self, model_path: str = "models/mobileclip_s2"):
        """
        Initialize MobileCLIP embedder.
        
        Args:
            model_path: Path to MobileCLIP model directory
        """
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        # Load model
        self.model, _, self.preprocess = mobileclip.create_model_and_transforms(
            'mobileclip_s2',
            pretrained=str(self.model_path / 'mobileclip_s2.pt')
        )
        self.model.eval()
        
        # Set device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)
    
    def embed(self, img_bgr: np.ndarray) -> np.ndarray:
        """
        Generate embedding from BGR image array (OpenCV format).
        
        Args:
            img_bgr: BGR image array (H x W x 3)
            
        Returns:
            512-dimensional embedding vector
        """
        # Convert OpenCV BGR -> RGB PIL image
        img = Image.fromarray(img_bgr[:, :, ::-1])
        
        # Preprocess
        img_tensor = self.preprocess(img).unsqueeze(0).to(self.device)
        
        # Encode
        with torch.no_grad():
            embedding = self.model.encode_image(img_tensor)
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)
        
        return embedding.cpu().numpy()[0]
    
    def encode_image(self, image_path: str) -> np.ndarray:
        """
        Generate embedding for a single image.
        
        Args:
            image_path: Path to image file
            
        Returns:
            512-dimensional embedding vector
        """
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            embedding = self.model.encode_image(image_tensor)
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)
        
        return embedding.cpu().numpy()[0]
    
    def encode_images_batch(self, image_paths: list) -> np.ndarray:
        """
        Generate embeddings for multiple images.
        
        Args:
            image_paths: List of image file paths
            
        Returns:
            Array of embeddings (N x 512)
        """
        images = []
        for path in image_paths:
            img = Image.open(path).convert('RGB')
            img_tensor = self.preprocess(img)
            images.append(img_tensor)
        
        batch = torch.stack(images).to(self.device)
        
        with torch.no_grad():
            embeddings = self.model.encode_image(batch)
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
        
        return embeddings.cpu().numpy()
