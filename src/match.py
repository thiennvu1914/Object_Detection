import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def find_best_match(query_vector, db_vectors, db_labels, threshold=0.5):
    """
    Tìm label khớp nhất với query vector trong database
    
    Args:
        query_vector: Vector embedding của crop cần classify (shape: (1, dim) hoặc (dim,))
        db_vectors: Array chứa tất cả vectors trong database (shape: (n, dim))
        db_labels: List các labels tương ứng với db_vectors
        threshold: Ngưỡng similarity tối thiểu để accept match (default: 0.5)
    
    Returns:
        (label, score): Tuple gồm label khớp nhất và similarity score
                        Trả về ("unknown", 0.0) nếu không có match tốt
    """
    # Đảm bảo query_vector có shape (1, dim)
    if query_vector.ndim == 1:
        query_vector = query_vector.reshape(1, -1)
    
    # Tính cosine similarity giữa query và tất cả vectors trong DB
    similarities = cosine_similarity(query_vector, db_vectors)[0]
    
    # Tìm index có similarity cao nhất
    best_idx = np.argmax(similarities)
    best_score = similarities[best_idx]
    
    # Kiểm tra threshold
    if best_score < threshold:
        return "unknown", float(best_score)
    
    best_label = db_labels[best_idx]
    
    return best_label, float(best_score)


def find_top_k_matches(query_vector, db_vectors, db_labels, k=3, threshold=0.5):
    """
    Tìm top-k labels khớp nhất với query vector
    
    Args:
        query_vector: Vector embedding của crop cần classify
        db_vectors: Array chứa tất cả vectors trong database
        db_labels: List các labels tương ứng với db_vectors
        k: Số lượng matches trả về (default: 3)
        threshold: Ngưỡng similarity tối thiểu
    
    Returns:
        List[(label, score)]: List các tuple (label, score) được sắp xếp theo score giảm dần
    """
    # Đảm bảo query_vector có shape (1, dim)
    if query_vector.ndim == 1:
        query_vector = query_vector.reshape(1, -1)
    
    # Tính cosine similarity
    similarities = cosine_similarity(query_vector, db_vectors)[0]
    
    # Lấy top-k indices
    top_k_indices = np.argsort(similarities)[-k:][::-1]
    
    # Tạo list kết quả
    results = []
    for idx in top_k_indices:
        score = similarities[idx]
        if score >= threshold:
            label = db_labels[idx]
            results.append((label, float(score)))
    
    return results


def aggregate_matches(matches, method='voting'):
    """
    Aggregate nhiều matches để đưa ra quyết định cuối cùng
    
    Args:
        matches: List các (label, score) từ find_top_k_matches
        method: 'voting' (đếm votes) hoặc 'weighted' (trung bình có trọng số)
    
    Returns:
        (final_label, confidence): Label cuối cùng và độ tin cậy
    """
    if not matches:
        return "unknown", 0.0
    
    if method == 'voting':
        # Đếm số lần xuất hiện của mỗi label
        from collections import Counter
        labels = [label for label, _ in matches]
        counter = Counter(labels)
        most_common = counter.most_common(1)[0]
        final_label = most_common[0]
        confidence = most_common[1] / len(matches)
        
        return final_label, confidence
    
    elif method == 'weighted':
        # Trung bình có trọng số theo score
        label_scores = {}
        for label, score in matches:
            if label not in label_scores:
                label_scores[label] = []
            label_scores[label].append(score)
        
        # Tính mean score cho mỗi label
        label_avg_scores = {
            label: np.mean(scores) 
            for label, scores in label_scores.items()
        }
        
        # Lấy label có avg score cao nhất
        final_label = max(label_avg_scores, key=label_avg_scores.get)
        confidence = label_avg_scores[final_label]
        
        return final_label, float(confidence)
    
    else:
        raise ValueError(f"Unknown aggregation method: {method}")
