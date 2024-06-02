import numpy as np
import jax
import jax.numpy as jnp
import flax

from flax import linen as nn


class sample_NN(nn.Module):

    @nn.compact
    def _call():
        x = 1 +1        

print("hello")