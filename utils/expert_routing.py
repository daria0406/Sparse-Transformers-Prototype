import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Any
import pandas as pd

class ExpertRouter:
    """
    Simulates expert routing mechanisms for Mixture-of-Experts (MoE) models.
    This is a demonstration class for educational purposes.
    """
    
    def __init__(self, num_experts: int, input_dim: int = 768, top_k: int = 2):
        self.num_experts = num_experts
        self.input_dim = input_dim
        self.top_k = top_k
        
        # Initialize routing weights (simulated)
        self.routing_weights = np.random.randn(input_dim, num_experts) * 0.1
        self.expert_specializations = self._initialize_expert_specializations()
        
    def _initialize_expert_specializations(self) -> Dict[int, str]:
        """Initialize expert specializations for demonstration."""
        specializations = [
            "Factual Verification", "Opinion Detection", "Context Analysis", "Temporal Reasoning",
            "Numerical Verification", "Entity Recognition", "Causal Reasoning", "Contradiction Detection",
            "Source Credibility", "Semantic Consistency", "Logical Inference", "Common Sense",
            "Scientific Claims", "Historical Facts", "Statistical Analysis", "Linguistic Patterns"
        ]
        
        return {i: specializations[i % len(specializations)] for i in range(self.num_experts)}
    
    def route(self, input_representation: np.ndarray, temperature: float = 1.0) -> Dict[str, Any]:
        """
        Route input to appropriate experts.
        
        Args:
            input_representation: Input vector representation
            temperature: Temperature for softmax routing
            
        Returns:
            Dictionary containing routing information
        """
        # Compute routing logits
        routing_logits = np.dot(input_representation, self.routing_weights)
        
        # Apply temperature scaling
        routing_logits = routing_logits / temperature
        
        # Compute routing probabilities
        routing_probs = self._softmax(routing_logits)
        
        # Select top-k experts
        top_k_indices = np.argsort(routing_probs)[-self.top_k:]
        top_k_probs = routing_probs[top_k_indices]
        
        # Normalize top-k probabilities
        top_k_probs = top_k_probs / np.sum(top_k_probs)
        
        return {
            "all_probs": routing_probs,
            "top_k_indices": top_k_indices,
            "top_k_probs": top_k_probs,
            "selected_experts": [self.expert_specializations[i] for i in top_k_indices],
            "routing_entropy": self._compute_entropy(routing_probs),
            "load_balance_loss": self._compute_load_balance_loss(routing_probs)
        }
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Compute softmax probabilities."""
        exp_x = np.exp(x - np.max(x))  # Numerical stability
        return exp_x / np.sum(exp_x)
    
    def _compute_entropy(self, probs: np.ndarray) -> float:
        """Compute entropy of routing distribution."""
        # Add small epsilon to avoid log(0)
        epsilon = 1e-12
        return -np.sum(probs * np.log(probs + epsilon))
    
    def _compute_load_balance_loss(self, probs: np.ndarray) -> float:
        """Compute load balancing loss to encourage uniform expert usage."""
        uniform_prob = 1.0 / self.num_experts
        return np.sum((probs - uniform_prob) ** 2)
    
    def analyze_routing_patterns(self, input_types: List[str], num_samples: int = 100) -> pd.DataFrame:
        """
        Analyze routing patterns for different input types.
        
        Args:
            input_types: List of input type descriptions
            num_samples: Number of samples to generate for each input type
            
        Returns:
            DataFrame with routing analysis
        """
        results = []
        
        for input_type in input_types:
            # Generate synthetic input representations for this type
            type_representations = self._generate_type_specific_representation(input_type, num_samples)
            
            for rep in type_representations:
                routing_result = self.route(rep)
                
                for expert_idx, prob in enumerate(routing_result["all_probs"]):
                    results.append({
                        "input_type": input_type,
                        "expert_idx": expert_idx,
                        "expert_name": self.expert_specializations[expert_idx],
                        "routing_prob": prob,
                        "is_selected": expert_idx in routing_result["top_k_indices"]
                    })
        
        return pd.DataFrame(results)
    
    def _generate_type_specific_representation(self, input_type: str, num_samples: int) -> np.ndarray:
        """Generate synthetic input representations based on input type."""
        # Create type-specific biases for demonstration
        type_biases = {
            "Scientific claim about climate change": np.array([0.8, 0.2, 0.6, 0.1, 0.9, 0.3, 0.7, 0.2]),
            "Political statement with statistics": np.array([0.6, 0.9, 0.8, 0.3, 0.8, 0.2, 0.5, 0.7]),
            "Historical fact about World War II": np.array([0.9, 0.1, 0.4, 0.8, 0.2, 0.6, 0.3, 0.1]),
            "Opinion about social media impact": np.array([0.2, 0.9, 0.7, 0.4, 0.1, 0.3, 0.6, 0.5]),
            "Medical claim about treatment": np.array([0.9, 0.1, 0.3, 0.2, 0.8, 0.4, 0.6, 0.3]),
            "Economic prediction": np.array([0.4, 0.6, 0.8, 0.5, 0.7, 0.2, 0.3, 0.9]),
            "Celebrity gossip": np.array([0.1, 0.8, 0.2, 0.3, 0.1, 0.9, 0.2, 0.4]),
            "Mathematical theorem": np.array([0.8, 0.1, 0.2, 0.1, 0.9, 0.1, 0.8, 0.2])
        }
        
        # Get bias pattern for this input type (or default)
        bias = type_biases.get(input_type, np.random.random(8))
        
        # Extend bias to match input dimension
        full_bias = np.tile(bias, (self.input_dim // len(bias) + 1))[:self.input_dim]
        
        # Generate samples with this bias
        representations = []
        for _ in range(num_samples):
            noise = np.random.randn(self.input_dim) * 0.1
            rep = full_bias + noise
            representations.append(rep)
        
        return np.array(representations)

def simulate_expert_routing(input_types: List[str], expert_names: List[str], 
                          top_k: int = 2, temperature: float = 1.0) -> Dict[str, Dict[str, float]]:
    """
    Simulate expert routing for visualization purposes.
    
    Args:
        input_types: List of input type descriptions
        expert_names: List of expert names/specializations
        top_k: Number of top experts to select
        temperature: Routing temperature
        
    Returns:
        Dictionary mapping input types to expert routing probabilities
    """
    num_experts = len(expert_names)
    router = ExpertRouter(num_experts, top_k=top_k)
    router.expert_specializations = {i: name for i, name in enumerate(expert_names)}
    
    routing_results = {}
    
    for input_type in input_types:
        # Generate a representative input vector for this type
        input_rep = router._generate_type_specific_representation(input_type, 1)[0]
        
        # Get routing probabilities
        routing_result = router.route(input_rep, temperature)
        
        # Create mapping from expert names to probabilities
        expert_probs = {}
        for i, expert_name in enumerate(expert_names):
            expert_probs[expert_name] = routing_result["all_probs"][i]
        
        routing_results[input_type] = expert_probs
    
    return routing_results

class ExpertLoadBalancer:
    """Manages load balancing across experts during training."""
    
    def __init__(self, num_experts: int, target_load: float = None):
        self.num_experts = num_experts
        self.target_load = target_load or (1.0 / num_experts)
        self.expert_loads = np.zeros(num_experts)
        self.total_samples = 0
        
    def update_loads(self, expert_assignments: np.ndarray):
        """Update expert load statistics."""
        for expert_idx in expert_assignments:
            self.expert_loads[expert_idx] += 1
        self.total_samples += len(expert_assignments)
    
    def get_current_loads(self) -> Dict[str, float]:
        """Get current load distribution."""
        if self.total_samples == 0:
            return {"loads": np.zeros(self.num_experts), "balance_score": 1.0}
        
        normalized_loads = self.expert_loads / self.total_samples
        balance_score = 1.0 - np.std(normalized_loads)
        
        return {
            "loads": normalized_loads,
            "balance_score": max(0.0, balance_score),
            "imbalance": np.max(normalized_loads) - np.min(normalized_loads)
        }
    
    def compute_auxiliary_loss(self) -> float:
        """Compute auxiliary loss to encourage load balancing."""
        current_loads = self.get_current_loads()["loads"]
        target_loads = np.full(self.num_experts, self.target_load)
        return np.mean((current_loads - target_loads) ** 2)

class ExpertSpecializationAnalyzer:
    """Analyzes expert specialization patterns."""
    
    def __init__(self, expert_names: List[str]):
        self.expert_names = expert_names
        self.num_experts = len(expert_names)
        
    def analyze_specialization_emergence(self, epochs: int = 100) -> Dict[str, np.ndarray]:
        """
        Simulate how expert specialization emerges during training.
        
        Args:
            epochs: Number of training epochs to simulate
            
        Returns:
            Dictionary with specialization curves for each expert
        """
        specialization_curves = {}
        
        for i, expert_name in enumerate(self.expert_names):
            # Different experts specialize at different rates
            rate = np.random.uniform(0.01, 0.1)
            saturation = np.random.uniform(0.7, 0.95)
            
            # Sigmoid-like specialization curve
            x = np.arange(epochs)
            curve = saturation * (1 - np.exp(-rate * x))
            
            # Add some noise
            noise = np.random.normal(0, 0.02, epochs)
            curve = np.clip(curve + noise, 0, 1)
            
            specialization_curves[expert_name] = curve
        
        return specialization_curves
    
    def compute_expert_similarity(self, routing_patterns: Dict[str, Dict[str, float]]) -> np.ndarray:
        """
        Compute similarity matrix between experts based on their routing patterns.
        
        Args:
            routing_patterns: Dictionary mapping input types to expert probabilities
            
        Returns:
            Similarity matrix between experts
        """
        # Extract expert probability vectors for each input type
        expert_vectors = {expert: [] for expert in self.expert_names}
        
        for input_type, expert_probs in routing_patterns.items():
            for expert, prob in expert_probs.items():
                expert_vectors[expert].append(prob)
        
        # Compute cosine similarity between experts
        similarity_matrix = np.zeros((self.num_experts, self.num_experts))
        
        for i, expert_i in enumerate(self.expert_names):
            for j, expert_j in enumerate(self.expert_names):
                vec_i = np.array(expert_vectors[expert_i])
                vec_j = np.array(expert_vectors[expert_j])
                
                # Cosine similarity
                dot_product = np.dot(vec_i, vec_j)
                norm_i = np.linalg.norm(vec_i)
                norm_j = np.linalg.norm(vec_j)
                
                if norm_i > 0 and norm_j > 0:
                    similarity_matrix[i, j] = dot_product / (norm_i * norm_j)
                else:
                    similarity_matrix[i, j] = 0.0
        
        return similarity_matrix
    
    def identify_expert_clusters(self, similarity_matrix: np.ndarray, threshold: float = 0.7) -> List[List[str]]:
        """
        Identify clusters of similar experts.
        
        Args:
            similarity_matrix: Expert similarity matrix
            threshold: Similarity threshold for clustering
            
        Returns:
            List of expert clusters
        """
        clusters = []
        assigned = set()
        
        for i in range(self.num_experts):
            if i in assigned:
                continue
                
            cluster = [self.expert_names[i]]
            assigned.add(i)
            
            for j in range(i + 1, self.num_experts):
                if j not in assigned and similarity_matrix[i, j] > threshold:
                    cluster.append(self.expert_names[j])
                    assigned.add(j)
            
            if len(cluster) > 0:
                clusters.append(cluster)
        
        return clusters

def generate_expert_performance_data(expert_names: List[str], num_tasks: int = 100) -> pd.DataFrame:
    """
    Generate synthetic expert performance data for visualization.
    
    Args:
        expert_names: List of expert names
        num_tasks: Number of tasks to simulate
        
    Returns:
        DataFrame with expert performance data
    """
    performance_data = []
    
    task_types = [
        "Fact Verification", "Opinion Classification", "Context Understanding",
        "Temporal Reasoning", "Numerical Analysis", "Entity Recognition",
        "Logical Inference", "Consistency Check"
    ]
    
    for task_id in range(num_tasks):
        task_type = np.random.choice(task_types)
        
        for expert_name in expert_names:
            # Simulate expert performance based on specialization
            base_performance = np.random.beta(6, 2)  # Generally good performance
            
            # Add specialization bonus
            if any(keyword in expert_name.lower() for keyword in task_type.lower().split()):
                specialization_bonus = np.random.uniform(0.05, 0.15)
                base_performance = min(1.0, base_performance + specialization_bonus)
            
            performance_data.append({
                "task_id": task_id,
                "task_type": task_type,
                "expert_name": expert_name,
                "performance": base_performance,
                "confidence": np.random.beta(4, 2),
                "processing_time": np.random.lognormal(0, 0.5)
            })
    
    return pd.DataFrame(performance_data)

class ExpertRouterDemo:
    """
    Demonstration class for expert routing mechanisms.
    Educational implementation for concept visualization.
    """
    
    def __init__(self, input_dim: int = 512, num_experts: int = 8, expert_capacity: int = 64):
        self.input_dim = input_dim
        self.num_experts = num_experts
        self.expert_capacity = expert_capacity
        
        # Initialize router
        self.router = ExpertRouter(num_experts=num_experts, input_dim=input_dim)
        
    def demonstrate_routing(self, batch_size: int = 100) -> Dict[str, Any]:
        """
        Demonstrate expert routing with sample data.
        
        Args:
            batch_size: Number of samples to route
            
        Returns:
            Dictionary with routing results
        """
        # Generate sample input representations
        np.random.seed(42)
        sample_inputs = np.random.randn(batch_size, self.input_dim)
        
        # Route all samples
        routing_results = []
        expert_loads = np.zeros(self.num_experts)
        routing_scores = np.zeros((batch_size, self.num_experts))
        
        for i, input_rep in enumerate(sample_inputs):
            result = self.router.route(input_rep)
            routing_results.append(result)
            routing_scores[i] = result['routing_probabilities']
            
            # Count expert loads
            for expert_id in result['selected_experts']:
                expert_loads[expert_id] += 1
        
        return {
            'routing_results': routing_results,
            'expert_loads': expert_loads,
            'routing_scores': routing_scores,
            'load_balance_coefficient': self._calculate_load_balance(expert_loads)
        }
    
    def analyze_expert_specialization(self, num_samples: int = 500) -> Dict[int, Dict[str, Any]]:
        """
        Analyze expert specialization patterns.
        
        Args:
            num_samples: Number of samples for analysis
            
        Returns:
            Dictionary with specialization analysis per expert
        """
        # Generate diverse sample types
        np.random.seed(42)
        sample_types = ['factual', 'opinion', 'temporal', 'numerical', 'causal']
        
        expert_assignments = {i: [] for i in range(self.num_experts)}
        
        for _ in range(num_samples):
            # Simulate different types of content
            sample_type = np.random.choice(sample_types)
            input_rep = self._generate_typed_input(sample_type)
            
            result = self.router.route(input_rep)
            for expert_id in result['selected_experts']:
                expert_assignments[expert_id].append(sample_type)
        
        # Analyze specialization
        specialization_analysis = {}
        for expert_id in range(self.num_experts):
            assignments = expert_assignments[expert_id]
            if assignments:
                type_counts = pd.Series(assignments).value_counts()
                specialization_score = self._calculate_specialization_score(type_counts)
                dominant_type = type_counts.index[0] if len(type_counts) > 0 else 'none'
            else:
                specialization_score = 0.0
                dominant_type = 'none'
                type_counts = pd.Series()
            
            specialization_analysis[expert_id] = {
                'specialization_score': specialization_score,
                'dominant_type': dominant_type,
                'type_distribution': type_counts.to_dict(),
                'total_assignments': len(assignments),
                'expert_name': self.router.expert_specializations.get(expert_id, f'Expert_{expert_id}')
            }
        
        return specialization_analysis
    
    def _calculate_load_balance(self, expert_loads: np.ndarray) -> float:
        """Calculate load balance coefficient (1.0 = perfectly balanced)"""
        if np.sum(expert_loads) == 0:
            return 1.0
        
        expected_load = np.mean(expert_loads)
        if expected_load == 0:
            return 1.0
        
        variance = np.var(expert_loads)
        return max(0.0, 1.0 - (variance / (expected_load ** 2)))
    
    def _generate_typed_input(self, sample_type: str) -> np.ndarray:
        """Generate input representation biased toward specific content type"""
        base_input = np.random.randn(self.input_dim) * 0.1
        
        # Add type-specific bias
        type_biases = {
            'factual': np.array([1.0, 0.5, -0.3, 0.8] * (self.input_dim // 4 + 1))[:self.input_dim],
            'opinion': np.array([-0.5, 1.0, 0.7, -0.2] * (self.input_dim // 4 + 1))[:self.input_dim],
            'temporal': np.array([0.3, -0.8, 1.0, 0.4] * (self.input_dim // 4 + 1))[:self.input_dim],
            'numerical': np.array([0.9, 0.2, -0.6, 1.0] * (self.input_dim // 4 + 1))[:self.input_dim],
            'causal': np.array([-0.7, 0.9, 0.5, -1.0] * (self.input_dim // 4 + 1))[:self.input_dim]
        }
        
        return base_input + 0.3 * type_biases.get(sample_type, np.zeros(self.input_dim))
    
    def _calculate_specialization_score(self, type_counts: pd.Series) -> float:
        """Calculate how specialized an expert is (higher = more specialized)"""
        if len(type_counts) <= 1:
            return 1.0
        
        # Calculate entropy-based specialization score
        total = type_counts.sum()
        probabilities = type_counts / total
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
        max_entropy = np.log2(len(type_counts))
        
        # Return normalized inverse entropy (higher = more specialized)
        return max(0.0, 1.0 - (entropy / max_entropy))
