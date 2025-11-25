"""
Food Classifier Module
======================
Classify food items using embedding similarity.
"""
from pathlib import Path
import numpy as np
from typing import Dict, List, Tuple


class FoodClassifier:
    """Classify food items based on embedding similarity"""
    
    def __init__(self, reference_embeddings: Dict[str, np.ndarray]):
        """
        Initialize classifier with reference embeddings.
        
        Args:
            reference_embeddings: Dict mapping class names to embedding arrays
        """
        self.reference_embeddings = reference_embeddings
        self.classes = list(reference_embeddings.keys())
    
    def classify(self, embedding: np.ndarray, threshold: float = 0.15) -> Tuple[str, float]:
        """
        Classify a single embedding.
        
        Args:
            embedding: 512-dim embedding vector
            threshold: Minimum similarity threshold (default: 0.15)
            
        Returns:
            Tuple of (predicted_class, similarity_score)
            Returns ('unknown', score) if score < threshold
        """
        best_class = "unknown"
        best_similarity = -1.0
        
        for class_name, ref_embeddings in self.reference_embeddings.items():
            # Calculate cosine similarity with all reference embeddings
            similarities = np.dot(ref_embeddings, embedding)
            avg_similarity = np.mean(similarities)
            
            if avg_similarity > best_similarity:
                best_similarity = avg_similarity
                best_class = class_name
        
        if best_similarity < threshold:
            return "unknown", float(best_similarity)
            
        return best_class, float(best_similarity)
    
    def classify_batch(self, embeddings: np.ndarray, threshold: float = 0.15) -> List[Tuple[str, float]]:
        """
        Classify multiple embeddings.
        
        Args:
            embeddings: Array of embeddings (N x 512)
            threshold: Minimum similarity threshold
            
        Returns:
            List of (predicted_class, similarity_score) tuples
        """
        results = []
        for embedding in embeddings:
            result = self.classify(embedding, threshold=threshold)
            results.append(result)
        return results
    
    def get_top_k(self, embedding: np.ndarray, k: int = 3) -> List[Tuple[str, float]]:
        """
        Get top-k predictions for an embedding.
        
        Args:
            embedding: 512-dim embedding vector
            k: Number of top predictions to return
            
        Returns:
            List of (class_name, similarity_score) sorted by score
        """
        scores = []
        for class_name, ref_embeddings in self.reference_embeddings.items():
            similarities = np.dot(ref_embeddings, embedding)
            avg_similarity = np.mean(similarities)
            scores.append((class_name, float(avg_similarity)))
        
        # Sort by similarity descending
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]
