import numpy as np
import torch
from typing import Tuple, Dict, Any

def generate_attention_matrix(seq_len: int, pattern_type: str = "dense", sparsity: float = 0.0, **kwargs) -> np.ndarray:
    """
    Generate different types of attention patterns for visualization.
    
    Args:
        seq_len: Sequence length
        pattern_type: Type of attention pattern ('dense', 'sparse', 'local', 'strided', 'random', 'structured', 'custom')
        sparsity: Sparsity level (0.0 = dense, 1.0 = completely sparse)
        **kwargs: Additional parameters for specific patterns
    
    Returns:
        Attention matrix of shape (seq_len, seq_len)
    """
    
    if pattern_type == "dense":
        # Full dense attention
        matrix = np.random.uniform(0.1, 1.0, (seq_len, seq_len))
        # Make it slightly lower triangular to simulate causal attention
        for i in range(seq_len):
            for j in range(i+1, seq_len):
                matrix[i, j] *= 0.3
    
    elif pattern_type == "sparse":
        # Random sparse attention
        matrix = np.zeros((seq_len, seq_len))
        num_connections = int(seq_len * seq_len * (1 - sparsity))
        
        # Add random connections
        indices = np.random.choice(seq_len * seq_len, num_connections, replace=False)
        for idx in indices:
            i, j = divmod(idx, seq_len)
            matrix[i, j] = np.random.uniform(0.1, 1.0)
    
    elif pattern_type == "local":
        # Local attention (sliding window)
        window_size = kwargs.get('window_size', max(1, seq_len // 8))
        matrix = np.zeros((seq_len, seq_len))
        
        for i in range(seq_len):
            start = max(0, i - window_size)
            end = min(seq_len, i + window_size + 1)
            for j in range(start, end):
                matrix[i, j] = np.random.uniform(0.5, 1.0)
    
    elif pattern_type == "strided":
        # Strided attention
        stride = kwargs.get('stride', max(1, seq_len // 16))
        matrix = np.zeros((seq_len, seq_len))
        
        for i in range(seq_len):
            # Local connections
            for j in range(max(0, i-2), min(seq_len, i+3)):
                matrix[i, j] = np.random.uniform(0.3, 1.0)
            
            # Strided connections
            for k in range(1, seq_len // stride):
                j = i + k * stride
                if j < seq_len:
                    matrix[i, j] = np.random.uniform(0.2, 0.8)
                j = i - k * stride
                if j >= 0:
                    matrix[i, j] = np.random.uniform(0.2, 0.8)
    
    elif pattern_type == "random":
        # Random sparse pattern
        matrix = np.random.uniform(0, 1, (seq_len, seq_len))
        mask = np.random.random((seq_len, seq_len)) < (1 - sparsity)
        matrix = matrix * mask
    
    elif pattern_type == "structured":
        # Block-sparse attention
        block_size = kwargs.get('block_size', max(1, seq_len // 8))
        matrix = np.zeros((seq_len, seq_len))
        
        num_blocks = seq_len // block_size
        for i in range(num_blocks):
            for j in range(num_blocks):
                if np.random.random() > sparsity:
                    start_i, end_i = i * block_size, (i + 1) * block_size
                    start_j, end_j = j * block_size, (j + 1) * block_size
                    matrix[start_i:end_i, start_j:end_j] = np.random.uniform(0.1, 1.0, (block_size, block_size))
    
    elif pattern_type == "custom":
        # Custom pattern with multiple components
        window_size = kwargs.get('window_size', seq_len // 8)
        stride = kwargs.get('stride', seq_len // 16)
        random_ratio = kwargs.get('random_ratio', 0.1)
        
        matrix = np.zeros((seq_len, seq_len))
        
        # Local attention
        for i in range(seq_len):
            start = max(0, i - window_size)
            end = min(seq_len, i + window_size + 1)
            for j in range(start, end):
                matrix[i, j] = np.random.uniform(0.4, 1.0)
        
        # Strided attention
        for i in range(seq_len):
            for k in range(1, seq_len // stride):
                j = i + k * stride
                if j < seq_len:
                    matrix[i, j] = max(matrix[i, j], np.random.uniform(0.2, 0.6))
        
        # Random connections
        num_random = int(seq_len * seq_len * random_ratio)
        indices = np.random.choice(seq_len * seq_len, num_random, replace=False)
        for idx in indices:
            i, j = divmod(idx, seq_len)
            matrix[i, j] = max(matrix[i, j], np.random.uniform(0.1, 0.5))
    
    else:
        raise ValueError(f"Unknown pattern type: {pattern_type}")
    
    # Normalize rows to sum to 1 (attention weights should sum to 1)
    row_sums = matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    matrix = matrix / row_sums
    
    return matrix

def calculate_sparsity_metrics(attention_matrix: np.ndarray) -> Dict[str, Any]:
    """
    Calculate various sparsity and efficiency metrics for an attention matrix.
    
    Args:
        attention_matrix: Attention matrix of shape (seq_len, seq_len)
    
    Returns:
        Dictionary containing sparsity metrics
    """
    seq_len = attention_matrix.shape[0]
    total_elements = seq_len * seq_len
    
    # Basic sparsity
    non_zero_elements = np.count_nonzero(attention_matrix)
    sparsity_ratio = 1 - (non_zero_elements / total_elements)
    
    # Memory usage (assuming float32)
    memory_mb = (non_zero_elements * 4) / (1024 * 1024)  # 4 bytes per float32
    
    # Computational complexity
    effective_complexity = non_zero_elements
    
    # Pattern analysis
    # Local connectivity (connections within a small window)
    local_window = max(1, seq_len // 16)
    local_connections = 0
    for i in range(seq_len):
        for j in range(max(0, i - local_window), min(seq_len, i + local_window + 1)):
            if attention_matrix[i, j] > 0:
                local_connections += 1
    
    local_ratio = local_connections / non_zero_elements if non_zero_elements > 0 else 0
    
    # Long-range connections
    long_range_connections = 0
    long_range_threshold = seq_len // 4
    for i in range(seq_len):
        for j in range(seq_len):
            if attention_matrix[i, j] > 0 and abs(i - j) > long_range_threshold:
                long_range_connections += 1
    
    long_range_ratio = long_range_connections / non_zero_elements if non_zero_elements > 0 else 0
    
    # Pattern regularity (how structured the pattern is)
    # Measure variance in row sparsity
    row_sparsities = [np.count_nonzero(row) / seq_len for row in attention_matrix]
    regularity_score = 1 - np.std(row_sparsities) if len(row_sparsities) > 0 else 0
    
    return {
        "non_zero_elements": non_zero_elements,
        "sparsity_ratio": sparsity_ratio,
        "memory_mb": memory_mb,
        "effective_complexity": effective_complexity,
        "local_ratio": local_ratio,
        "long_range_ratio": long_range_ratio,
        "regularity_score": regularity_score
    }

def compute_attention_efficiency(seq_len: int, sparsity: float) -> Dict[str, float]:
    """
    Compute theoretical efficiency gains from sparse attention.
    
    Args:
        seq_len: Sequence length
        sparsity: Sparsity level (0.0 to 1.0)
    
    Returns:
        Dictionary with efficiency metrics
    """
    dense_ops = seq_len ** 2
    sparse_ops = dense_ops * (1 - sparsity)
    
    # Memory savings
    dense_memory = seq_len ** 2 * 4  # 4 bytes per float32
    sparse_memory = sparse_ops * 4
    memory_reduction = dense_memory / sparse_memory if sparse_memory > 0 else float('inf')
    
    # Computational savings
    compute_reduction = dense_ops / sparse_ops if sparse_ops > 0 else float('inf')
    
    # Theoretical speedup (considers both memory and compute)
    theoretical_speedup = min(memory_reduction, compute_reduction) * 0.8  # 80% efficiency
    
    return {
        "memory_reduction": memory_reduction,
        "compute_reduction": compute_reduction,
        "theoretical_speedup": theoretical_speedup,
        "dense_ops": dense_ops,
        "sparse_ops": sparse_ops
    }

def visualize_attention_pattern(attention_matrix: np.ndarray, title: str = "Attention Pattern") -> Dict[str, Any]:
    """
    Prepare attention matrix data for visualization.
    
    Args:
        attention_matrix: Attention matrix to visualize
        title: Title for the visualization
    
    Returns:
        Dictionary with visualization data
    """
    seq_len = attention_matrix.shape[0]
    
    # Calculate key statistics
    metrics = calculate_sparsity_metrics(attention_matrix)
    
    # Prepare heatmap data
    heatmap_data = {
        "matrix": attention_matrix,
        "x_labels": [f"Pos_{i}" for i in range(seq_len)],
        "y_labels": [f"Pos_{i}" for i in range(seq_len)],
        "title": title
    }
    
    # Calculate attention statistics per position
    attention_stats = {
        "position": list(range(seq_len)),
        "total_attention": attention_matrix.sum(axis=1).tolist(),
        "num_connections": [np.count_nonzero(row) for row in attention_matrix],
        "max_attention": attention_matrix.max(axis=1).tolist(),
        "mean_attention": attention_matrix.mean(axis=1).tolist()
    }
    
    return {
        "heatmap_data": heatmap_data,
        "metrics": metrics,
        "attention_stats": attention_stats
    }

class AttentionPatternGenerator:
    """Class for generating and analyzing various attention patterns."""
    
    def __init__(self):
        self.patterns = {
            "dense": self._generate_dense,
            "sparse": self._generate_sparse,
            "local": self._generate_local,
            "strided": self._generate_strided,
            "block": self._generate_block,
            "random": self._generate_random
        }
    
    def generate(self, pattern_type: str, seq_len: int, **kwargs) -> np.ndarray:
        """Generate attention pattern of specified type."""
        if pattern_type not in self.patterns:
            raise ValueError(f"Unknown pattern type: {pattern_type}")
        
        return self.patterns[pattern_type](seq_len, **kwargs)
    
    def _generate_dense(self, seq_len: int, **kwargs) -> np.ndarray:
        """Generate dense attention pattern."""
        return generate_attention_matrix(seq_len, "dense", 0.0, **kwargs)
    
    def _generate_sparse(self, seq_len: int, sparsity: float = 0.8, **kwargs) -> np.ndarray:
        """Generate sparse attention pattern."""
        return generate_attention_matrix(seq_len, "sparse", sparsity, **kwargs)
    
    def _generate_local(self, seq_len: int, window_size: int = None, **kwargs) -> np.ndarray:
        """Generate local attention pattern."""
        if window_size is None:
            window_size = max(1, seq_len // 8)
        return generate_attention_matrix(seq_len, "local", 0.0, window_size=window_size, **kwargs)
    
    def _generate_strided(self, seq_len: int, stride: int = None, **kwargs) -> np.ndarray:
        """Generate strided attention pattern."""
        if stride is None:
            stride = max(1, seq_len // 16)
        return generate_attention_matrix(seq_len, "strided", 0.0, stride=stride, **kwargs)
    
    def _generate_block(self, seq_len: int, block_size: int = None, sparsity: float = 0.5, **kwargs) -> np.ndarray:
        """Generate block-sparse attention pattern."""
        if block_size is None:
            block_size = max(1, seq_len // 8)
        return generate_attention_matrix(seq_len, "structured", sparsity, block_size=block_size, **kwargs)
    
    def _generate_random(self, seq_len: int, sparsity: float = 0.7, **kwargs) -> np.ndarray:
        """Generate random sparse attention pattern."""
        return generate_attention_matrix(seq_len, "random", sparsity, **kwargs)
