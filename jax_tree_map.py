import jax
import jax.numpy as jnp
from jax.tree_util import tree_map

tree = {"a": [1, 2], "b": (3, 4), "c": jnp.array([5, 6])}

def add_one(x):
    return x + 1

new_tree = tree_map(add_one, tree)
print(new_tree)

