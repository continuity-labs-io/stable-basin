import logging
import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import torch

from src.echo.architecture.observer import MarkovBlanketObserver
from src.echo.architecture.hierarchy import PredictiveCodingGraph
from src.echo.primitives.ebm import PrecisionWeightedEBM

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# =====================================================================
# Estimator D: Model-Free Classifier (Arrow-of-Time Neural Network)
# =====================================================================
class ArrowOfTimeClassifier(eqx.Module):
    """
    Estimator D: A lightweight Arrow-of-Time neural network (MLP) 
    trained via binary cross-entropy to distinguish forward time-windows 
    from reversed time-windows, providing a pure non-Gaussian EP lower bound.
    """
    layers: list

    def __init__(self, seq_len: int, n_components: int, hidden_size: int, key: jax.random.PRNGKey):
        k1, k2, k3 = jax.random.split(key, 3)
        in_features = seq_len * n_components
        
        self.layers = [
            eqx.nn.Linear(in_features, hidden_size, key=k1),
            jax.nn.relu,
            eqx.nn.Linear(hidden_size, hidden_size // 2, key=k2),
            jax.nn.relu,
            eqx.nn.Linear(hidden_size // 2, 1, key=k3)
        ]

    def __call__(self, x: jax.Array) -> jax.Array:
        # x expected shape: (seq_len, n_components)
        x = x.flatten()
        for layer in self.layers:
            x = layer(x)
        # Returns a scalar logit
        return x

def binary_cross_entropy(logits: jax.Array, labels: jax.Array) -> jax.Array:
    """Computes binary cross entropy loss given logits and labels."""
    return optax.sigmoid_binary_cross_entropy(logits, labels).mean()


# =====================================================================
# Estimator E: Fitted NESS EBM (Predictive Coding Graph)
# =====================================================================
def build_estimator_E(key: jax.random.PRNGKey, n_pca_components: int) -> PredictiveCodingGraph:
    """
    Estimator E: Instantiate the PredictiveCodingGraph utilizing the PrecisionWeightedEBM.
    Critical constraint: use_blanket_topology=False must be passed because EEG components 
    are a unified sensor array, so the dissipative matrix Gamma must remain a full-rank, 
    unmasked positive-definite matrix to preserve the fluctuation-dissipation theorem.
    """
    k1, k2, k3 = jax.random.split(key, 3)
    
    # We assign the PCA components primarily as sensory variables
    # The internal and active variables can provide latent capacity
    d_internal_micro = 4
    d_sensory_micro = n_pca_components
    d_active_micro = 4
    d_external_micro = 4
    d_micro = d_internal_micro + d_sensory_micro + d_active_micro + d_external_micro
    
    d_internal_macro = 4
    d_sensory_macro = 4
    d_active_macro = 4
    d_external_macro = 4
    
    # Create micro and macro observers with use_blanket_topology=False
    micro = MarkovBlanketObserver(
        d_internal_micro, d_sensory_micro, d_active_micro, d_external_micro, 
        ebm_hidden_size=32, ebm_depth=2, n_steps=1, temperature=1.0, 
        key=k1, use_blanket_topology=False
    )
                                  
    macro = MarkovBlanketObserver(
        d_internal_macro, d_sensory_macro, d_active_macro, d_external_macro, 
        ebm_hidden_size=16, ebm_depth=2, n_steps=1, temperature=1.0, 
        key=k2, use_blanket_topology=False
    )
    
    # Overwrite the ebm with the PrecisionWeightedEBM
    micro = eqx.tree_at(
        lambda m: m.ebm, 
        micro, 
        PrecisionWeightedEBM(d_state=d_micro, hidden_size=32, depth=2, key=k3)
    )
    macro = eqx.tree_at(
        lambda m: m.ebm, 
        macro, 
        PrecisionWeightedEBM(d_state=macro.hull.d_state, hidden_size=16, depth=2, key=k3)
    )
        
    graph = PredictiveCodingGraph(micro, macro, n_steps=1, key=k3)
    return graph


def main():
    logger.info("Initializing Estimators D & E for Nonlinear EEG Entropy.")
    
    key = jax.random.PRNGKey(42)
    seq_len = 3000
    n_pca_components = 5
    
    # Instantiate Estimator D
    key, subkey = jax.random.split(key)
    estimator_D = ArrowOfTimeClassifier(seq_len=seq_len, n_components=n_pca_components, hidden_size=64, key=subkey)
    logger.info(f"Instantiated Estimator D: ArrowOfTimeClassifier with {(seq_len * n_pca_components)} input features.")
    
    # Instantiate Estimator E
    key, subkey = jax.random.split(key)
    estimator_E = build_estimator_E(subkey, n_pca_components=n_pca_components)
    logger.info("Instantiated Estimator E: PredictiveCodingGraph with PrecisionWeightedEBM and use_blanket_topology=False.")

if __name__ == "__main__":
    main()
