import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import math

class SparseTransformerDemo:
    """
    Demonstration class for sparse transformer concepts.
    This is for educational purposes and visualization, not full training.
    """
    
    def __init__(self, vocab_size: int = 50000, d_model: int = 512, 
                 n_heads: int = 8, n_layers: int = 6, max_seq_len: int = 1024,
                 sparsity_type: str = "sparse", sparsity_ratio: float = 0.8):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len
        self.sparsity_type = sparsity_type
        self.sparsity_ratio = sparsity_ratio
        
        # Initialize components for demonstration
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = self._create_positional_encoding()
        self.transformer_layers = self._create_transformer_layers()
        self.output_projection = nn.Linear(d_model, vocab_size)
        
    def _create_positional_encoding(self) -> torch.Tensor:
        """Create positional encoding matrix."""
        pe = torch.zeros(self.max_seq_len, self.d_model)
        position = torch.arange(0, self.max_seq_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * 
                           (-math.log(10000.0) / self.d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe.unsqueeze(0)
    
    def _create_transformer_layers(self) -> nn.ModuleList:
        """Create transformer layers with sparse attention."""
        layers = nn.ModuleList()
        
        for _ in range(self.n_layers):
            layer = SparseTransformerLayer(
                d_model=self.d_model,
                n_heads=self.n_heads,
                sparsity_type=self.sparsity_type,
                sparsity_ratio=self.sparsity_ratio
            )
            layers.append(layer)
        
        return layers
    
    def forward_demo(self, input_ids: torch.Tensor) -> Dict[str, Any]:
        """
        Demonstration forward pass with attention visualization.
        
        Args:
            input_ids: Input token IDs
            
        Returns:
            Dictionary with outputs and attention patterns
        """
        batch_size, seq_len = input_ids.shape
        
        # Embedding + positional encoding
        embeddings = self.embedding(input_ids)
        pos_encoding = self.positional_encoding[:, :seq_len, :]
        hidden_states = embeddings + pos_encoding
        
        # Store attention patterns from each layer
        attention_patterns = []
        layer_outputs = []
        
        for i, layer in enumerate(self.transformer_layers):
            layer_output = layer.forward_with_attention(hidden_states)
            hidden_states = layer_output['output']
            attention_patterns.append(layer_output['attention_weights'])
            layer_outputs.append(hidden_states.clone())
        
        # Final projection
        logits = self.output_projection(hidden_states)
        
        return {
            'logits': logits,
            'attention_patterns': attention_patterns,
            'layer_outputs': layer_outputs,
            'final_hidden_states': hidden_states
        }
    
    def analyze_sparsity(self, seq_len: int = 128) -> Dict[str, Any]:
        """
        Analyze sparsity patterns and efficiency gains.
        
        Args:
            seq_len: Sequence length for analysis
            
        Returns:
            Dictionary with sparsity analysis
        """
        # Generate sample attention pattern
        sample_attention = self._generate_sample_attention(seq_len)
        
        # Calculate sparsity metrics
        total_elements = seq_len * seq_len
        non_zero_elements = torch.count_nonzero(sample_attention).item()
        actual_sparsity = 1 - (non_zero_elements / total_elements)
        
        # Efficiency calculations
        dense_ops = seq_len ** 2 * self.d_model
        sparse_ops = non_zero_elements * self.d_model
        
        efficiency_gain = dense_ops / sparse_ops if sparse_ops > 0 else float('inf')
        memory_reduction = total_elements / non_zero_elements if non_zero_elements > 0 else float('inf')
        
        return {
            'target_sparsity': self.sparsity_ratio,
            'actual_sparsity': actual_sparsity,
            'non_zero_elements': non_zero_elements,
            'total_elements': total_elements,
            'efficiency_gain': efficiency_gain,
            'memory_reduction': memory_reduction,
            'theoretical_speedup': min(efficiency_gain, memory_reduction) * 0.8
        }
    
    def _generate_sample_attention(self, seq_len: int) -> torch.Tensor:
        """Generate sample attention pattern based on sparsity type."""
        if self.sparsity_type == "dense":
            attention = torch.rand(seq_len, seq_len)
        elif self.sparsity_type == "sparse":
            attention = torch.zeros(seq_len, seq_len)
            num_connections = int(seq_len * seq_len * (1 - self.sparsity_ratio))
            indices = torch.randperm(seq_len * seq_len)[:num_connections]
            attention.view(-1)[indices] = torch.rand(num_connections)
        elif self.sparsity_type == "local":
            attention = torch.zeros(seq_len, seq_len)
            window_size = max(1, int(seq_len * (1 - self.sparsity_ratio)))
            for i in range(seq_len):
                start = max(0, i - window_size // 2)
                end = min(seq_len, i + window_size // 2 + 1)
                attention[i, start:end] = torch.rand(end - start)
        elif self.sparsity_type == "strided":
            attention = torch.zeros(seq_len, seq_len)
            stride = max(1, int(1 / (1 - self.sparsity_ratio)))
            for i in range(seq_len):
                for j in range(0, seq_len, stride):
                    if abs(i - j) <= 2:  # Local connections
                        attention[i, j] = torch.rand(1)
        else:
            attention = torch.rand(seq_len, seq_len)
        
        # Normalize to valid attention weights
        attention = F.softmax(attention, dim=-1)
        return attention
    
    def compare_attention_patterns(self, seq_len: int = 64) -> Dict[str, torch.Tensor]:
        """Compare different attention patterns."""
        patterns = {}
        
        pattern_types = ["dense", "sparse", "local", "strided"]
        
        for pattern_type in pattern_types:
            # Temporarily change sparsity type
            original_type = self.sparsity_type
            self.sparsity_type = pattern_type
            
            patterns[pattern_type] = self._generate_sample_attention(seq_len)
            
            # Restore original type
            self.sparsity_type = original_type
        
        return patterns

class SparseTransformerLayer(nn.Module):
    """
    Sparse transformer layer with configurable attention patterns.
    """
    
    def __init__(self, d_model: int, n_heads: int, sparsity_type: str = "sparse",
                 sparsity_ratio: float = 0.8, d_ff: int = None):
        super().__init__()
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.sparsity_type = sparsity_type
        self.sparsity_ratio = sparsity_ratio
        self.d_ff = d_ff or 4 * d_model
        
        # Multi-head attention components
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        # Feed-forward network
        self.ff_1 = nn.Linear(d_model, self.d_ff)
        self.ff_2 = nn.Linear(self.d_ff, d_model)
        
        # Layer normalization
        self.ln_1 = nn.LayerNorm(d_model)
        self.ln_2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(0.1)
        
    def forward_with_attention(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass with attention weight extraction."""
        batch_size, seq_len, d_model = x.shape
        
        # Pre-layer norm
        x_norm = self.ln_1(x)
        
        # Multi-head attention
        q = self.q_proj(x_norm).view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x_norm).view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x_norm).view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        
        # Scaled dot-product attention with sparsity
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)
        
        # Apply sparsity pattern
        attention_weights = self._apply_sparsity_pattern(attention_scores, seq_len)
        attention_weights = F.softmax(attention_weights, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        attention_output = torch.matmul(attention_weights, v)
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model
        )
        
        # Output projection and residual connection
        attention_output = self.out_proj(attention_output)
        x = x + self.dropout(attention_output)
        
        # Feed-forward network
        x_norm_2 = self.ln_2(x)
        ff_output = self.ff_2(F.relu(self.ff_1(x_norm_2)))
        x = x + self.dropout(ff_output)
        
        return {
            'output': x,
            'attention_weights': attention_weights
        }
    
    def _apply_sparsity_pattern(self, attention_scores: torch.Tensor, seq_len: int) -> torch.Tensor:
        """Apply sparsity pattern to attention scores."""
        batch_size, n_heads, seq_len, seq_len = attention_scores.shape
        
        if self.sparsity_type == "dense":
            return attention_scores
        
        # Create sparsity mask
        mask = torch.zeros_like(attention_scores, dtype=torch.bool)
        
        if self.sparsity_type == "sparse":
            # Random sparse pattern
            num_connections = int(seq_len * seq_len * (1 - self.sparsity_ratio))
            for b in range(batch_size):
                for h in range(n_heads):
                    indices = torch.randperm(seq_len * seq_len)[:num_connections]
                    flat_mask = mask[b, h].view(-1)
                    flat_mask[indices] = True
        
        elif self.sparsity_type == "local":
            # Local attention window
            window_size = max(1, int(seq_len * (1 - self.sparsity_ratio)))
            for i in range(seq_len):
                start = max(0, i - window_size // 2)
                end = min(seq_len, i + window_size // 2 + 1)
                mask[:, :, i, start:end] = True
        
        elif self.sparsity_type == "strided":
            # Strided attention pattern
            stride = max(1, int(1 / (1 - self.sparsity_ratio)))
            for i in range(seq_len):
                # Local connections
                for j in range(max(0, i-2), min(seq_len, i+3)):
                    mask[:, :, i, j] = True
                # Strided connections
                for j in range(0, seq_len, stride):
                    mask[:, :, i, j] = True
        
        # Apply mask
        attention_scores = attention_scores.masked_fill(~mask, float('-inf'))
        
        return attention_scores

class TruthfulnessHead(nn.Module):
    """
    Specialized head for truthfulness classification.
    """
    
    def __init__(self, d_model: int, num_classes: int = 2, dropout: float = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.num_classes = num_classes
        
        # Multi-layer classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, num_classes)
        )
        
        # Confidence estimation
        self.confidence_head = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, 1),
            nn.Sigmoid()
        )
    
    def forward(self, hidden_states: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass through truthfulness head.
        
        Args:
            hidden_states: Hidden states from transformer
            attention_mask: Optional attention mask
            
        Returns:
            Dictionary with logits and confidence scores
        """
        # Pool hidden states (use [CLS] token or mean pooling)
        if attention_mask is not None:
            # Mean pooling with attention mask
            masked_hidden = hidden_states * attention_mask.unsqueeze(-1)
            pooled = masked_hidden.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
        else:
            # Simple mean pooling
            pooled = hidden_states.mean(dim=1)
        
        # Classification logits
        logits = self.classifier(pooled)
        
        # Confidence estimation
        confidence = self.confidence_head(pooled)
        
        return {
            'logits': logits,
            'confidence': confidence,
            'pooled_hidden': pooled
        }

class SparseTransformerConfig:
    """Configuration class for sparse transformer models."""
    
    def __init__(self, **kwargs):
        # Model architecture
        self.vocab_size = kwargs.get('vocab_size', 50000)
        self.d_model = kwargs.get('d_model', 512)
        self.n_heads = kwargs.get('n_heads', 8)
        self.n_layers = kwargs.get('n_layers', 6)
        self.d_ff = kwargs.get('d_ff', None)  # Default: 4 * d_model
        self.max_seq_len = kwargs.get('max_seq_len', 1024)
        
        # Sparsity configuration
        self.sparsity_type = kwargs.get('sparsity_type', 'sparse')
        self.sparsity_ratio = kwargs.get('sparsity_ratio', 0.8)
        self.window_size = kwargs.get('window_size', None)
        self.stride = kwargs.get('stride', None)
        
        # Training configuration
        self.dropout = kwargs.get('dropout', 0.1)
        self.layer_norm_eps = kwargs.get('layer_norm_eps', 1e-5)
        
        # Task-specific configuration
        self.num_truthfulness_classes = kwargs.get('num_truthfulness_classes', 2)
        self.use_confidence_estimation = kwargs.get('use_confidence_estimation', True)
        
        # Expert configuration (for MoE variants)
        self.use_experts = kwargs.get('use_experts', False)
        self.num_experts = kwargs.get('num_experts', 8)
        self.expert_top_k = kwargs.get('expert_top_k', 2)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'SparseTransformerConfig':
        """Create configuration from dictionary."""
        return cls(**config_dict)

def create_sparse_transformer_demo(config: Optional[SparseTransformerConfig] = None) -> SparseTransformerDemo:
    """
    Factory function to create a sparse transformer demo instance.
    
    Args:
        config: Optional configuration object
        
    Returns:
        SparseTransformerDemo instance
    """
    if config is None:
        config = SparseTransformerConfig()
    
    return SparseTransformerDemo(
        vocab_size=config.vocab_size,
        d_model=config.d_model,
        n_heads=config.n_heads,
        n_layers=config.n_layers,
        max_seq_len=config.max_seq_len,
        sparsity_type=config.sparsity_type,
        sparsity_ratio=config.sparsity_ratio
    )

def demonstrate_attention_patterns(seq_len: int = 64) -> Dict[str, np.ndarray]:
    """
    Demonstrate different attention patterns for educational purposes.
    
    Args:
        seq_len: Sequence length
        
    Returns:
        Dictionary mapping pattern names to attention matrices
    """
    patterns = {}
    
    # Dense attention
    dense_demo = SparseTransformerDemo(sparsity_type="dense")
    patterns['Dense'] = dense_demo._generate_sample_attention(seq_len).numpy()
    
    # Sparse attention
    sparse_demo = SparseTransformerDemo(sparsity_type="sparse", sparsity_ratio=0.8)
    patterns['Sparse (80%)'] = sparse_demo._generate_sample_attention(seq_len).numpy()
    
    # Local attention
    local_demo = SparseTransformerDemo(sparsity_type="local", sparsity_ratio=0.9)
    patterns['Local Window'] = local_demo._generate_sample_attention(seq_len).numpy()
    
    # Strided attention
    strided_demo = SparseTransformerDemo(sparsity_type="strided", sparsity_ratio=0.85)
    patterns['Strided'] = strided_demo._generate_sample_attention(seq_len).numpy()
    
    return patterns

def calculate_computational_savings(seq_lengths: List[int], sparsity_ratios: List[float]) -> pd.DataFrame:
    """
    Calculate computational savings for different sequence lengths and sparsity ratios.
    
    Args:
        seq_lengths: List of sequence lengths to analyze
        sparsity_ratios: List of sparsity ratios to analyze
        
    Returns:
        DataFrame with computational savings analysis
    """
    import pandas as pd
    
    results = []
    
    for seq_len in seq_lengths:
        for sparsity in sparsity_ratios:
            # Dense operations
            dense_ops = seq_len ** 2
            dense_memory = dense_ops * 4  # 4 bytes per float32
            
            # Sparse operations
            sparse_ops = dense_ops * (1 - sparsity)
            sparse_memory = sparse_ops * 4
            
            # Savings
            compute_saving = (dense_ops - sparse_ops) / dense_ops
            memory_saving = (dense_memory - sparse_memory) / dense_memory
            theoretical_speedup = dense_ops / sparse_ops if sparse_ops > 0 else float('inf')
            
            results.append({
                'sequence_length': seq_len,
                'sparsity_ratio': sparsity,
                'dense_operations': dense_ops,
                'sparse_operations': sparse_ops,
                'compute_saving_percent': compute_saving * 100,
                'memory_saving_percent': memory_saving * 100,
                'theoretical_speedup': theoretical_speedup
            })
    
    return pd.DataFrame(results)
