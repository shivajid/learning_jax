import jax
import jax.numpy as jnp

# Define a function to be executed in parallel
def my_function(x):
    return x ** 2 + 3 * x

# Create some input data
#data = jnp.arange(4)
data = jax.numpy.ones([4,4])

# Use pmap to apply the function to the data in parallel
result = jax.pmap(my_function)(data)

print(result)

