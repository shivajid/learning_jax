import jax
import jax.numpy as jnp
from jax.experimental import maps
from jax.experimental.pjit import pjit
from flax.training.train_state import TrainState

# TPU configuration
jax.config.update("jax_platform_name", "tpu")
jax.config.update("jax_xla_backend", "tpu_driver")
jax.config.update("jax_tpu_driver_version", "tpu_driver_nightly")

# Get TPU mesh
devices = jax.devices()
mesh = maps.Mesh(np.asarray(devices).reshape(2, 4), ("x", "y"))

# Generate input matrices (randomly for demonstration)
key = jax.random.PRNGKey(42)
mat1 = jax.random.normal(key, (1024, 1024))
mat2 = jax.random.normal(key, (1024, 1024))

# Shard input matrices
mat1_shard = pjit.with_sharding_constraint(mat1, ("x", "y"))
mat2_shard = pjit.with_sharding_constraint(mat2, ("x", "y"))

# Define matrix multiplication function
@pjit.pjit(in_axis_resources=(("x", "y"), ("x", "y")), out_axis_resources=("x", "y"))
def matmul_shard(x, y):
    return jnp.dot(x, y)

# Parallel matrix multiplication using pmap
result = pjit.pmap(matmul_shard)(mat1_shard, mat2_shard)

# Aggregate the result (if needed)
result = pjit.psum(result, ("x", "y"))

# Validate the result (compare against non-sharded computation)
expected_result = jnp.dot(mat1, mat2)
assert jnp.allclose(result, expected_result)

