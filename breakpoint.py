import numpy as np
import jax.numpy as jnp
import jax

MATRIX_DIM = 16384
A = jnp.ones([MATRIX_DIM, MATRIX_DIM])

#Check the devices
print(A.devices())

mesh =  jax.sharding.Mesh(jax.devices(), ("myaxis"))
p = jax.sharding.PartitionSpec("myaxis") 
sharding =  jax.sharding.NamedSharding(mesh,p)
breakpoint()



