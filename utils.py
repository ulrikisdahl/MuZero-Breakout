import importlib
import torch
import torch.nn as nn
import torch.nn.functional as F



class ScalarTransforms(nn.Module):
    """
    Class for representing scalar values as support distributions
    """
    def __init__(self, cfg: dict):
        super().__init__()
        self.epsilon = 0.001
        supports_min = cfg["supports_min"]
        supports_max = cfg["supports_max"]
        self.num_supports = cfg["num_supports"]
        self.device = cfg["device"]
        self.supports = torch.linspace(supports_min, supports_max, self.num_supports).to(self.device)

    def _invertible_transform_normal_to_compact(self, x):
        """ Maps the reward/value to a more compact representation to compress large values into the support representation range
        Obtain categorical representations of the reward/value targets equivalent to the output representations of the networks """
        return torch.sign(x) * (torch.sqrt(torch.abs(x) + 1) - 1 + self.epsilon * x)
        
    def _invertible_transform_compact_to_normal(self, x):
        """ Maps the reward/value back to the original representation """
        return torch.sign(x) * ((torch.abs(x) + (1-self.epsilon))**2 - 1)
        
    def supports_representation(self, target_value):
        """ Rewards and Values represented categorically with a possible range of [-300, 300]
        Original value x is represented as x = p_low * x_low + p_high * x_high (page: 14)
        1. Transform target scalar using invertible transformation to compress
        2. Map it to the support set using a linear combination of two adjacent supports
        3. Return a probability distribution over the supports
        
        Args:
            target_value (batch_size, K): Observed rewards or values at each step k in the sample
            
        Return:
            support_vector (batch_size, K, num_supports): A probability distribution over the supports
        """
        # Transform to compact representation
        target_transformed = self._invertible_transform_normal_to_compact(target_value)
        
        # Find the closest support indices
        lower_idx = torch.searchsorted(self.supports, target_transformed, right=True) - 1
        lower_idx = lower_idx.clamp(0, self.num_supports - 2)  # Fix 3: Ensure upper_idx doesn't go out of bounds
        upper_idx = lower_idx + 1
        
        # Get the supports
        lower_support = self.supports[lower_idx]
        upper_support = self.supports[upper_idx]
        
        # Compute linear combination coefficients
        p_low = (upper_support - target_transformed) / (upper_support - lower_support + 1e-10)
        p_high = 1 - p_low
        
        batch_size, k = target_value.shape
        support_vector = torch.zeros((batch_size, k, self.num_supports)).to(self.device)
        support_vector.scatter_(2, lower_idx.unsqueeze(-1), p_low.unsqueeze(-1))
        support_vector.scatter_(2, upper_idx.unsqueeze(-1), p_high.unsqueeze(-1))
        
        return support_vector
    
    def _softmax_expectation(self, softmax_distribution):
        """
        Computes the expectation of a softmax distribution over the supports

        Used for inference
        """
        return torch.sum(softmax_distribution * self.supports, dim=-1)

    def inverted_softmax_expectation(self, softmax_distribution):
        """
        First computes the expected value under the respective "softmax" distribution and subsequently inverts the scaling transformation
        """
        softmax_distribution = F.softmax(softmax_distribution, dim=-1)
        softmax_expectation = self._softmax_expectation(softmax_distribution)
        inverted_transform = self._invertible_transform_compact_to_normal(softmax_expectation)
        return inverted_transform


def get_class(module_name: str, class_name: str):
    """
    Dynamically imports and retreives classes

    Args:
        module_name (path): module location
        class_name (string): name of the class
    """
    try:
        module = importlib.import_module(module_name)
        return getattr(module, class_name)
    except:
        raise ImportError(f"Could not import module {module_name}")
    

def torch_activation_map(activation: str) -> nn.Module:
    """
    Returns the torch activation function based on the string input
    """
    return {
        "relu": nn.ReLU,
        "leaky_relu": nn.LeakyReLU,
        "silu": nn.SiLU,
        "gelu": nn.GELU
    }[activation]