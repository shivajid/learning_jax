import numpy as np
import jax
import jax.numpy as jnp

#Test with 
#MATRIX_DIM = 65536
#A = jnp.ones([MATRIX_DIM. MATRIX_DIM])

A = np.ones((65536,65536))

B = np.ones((65536,65536))
#Start the profiler
jax.profiler.start_trace("/tmp/profile_me")

#Row Sharding
mesh =  jax.sharding.Mesh(np.reshape(jax.devices(),(2,2)), ["myaxis1","myaxis2"])
p = jax.sharding.PartitionSpec("myaxis1","myaxis2") 
sharding =  jax.sharding.NamedSharding(mesh,p)


sharded_A = jax.device_put(A, sharding)
sharded_B = jax.device_put(B, sharding)

C = sharded_A + sharded_B
jax.profiler.stop_trace()

#Visualize Matrix A
jax.debug.visualize_array_sharding(sharded_A)

print ( sharded_A.addressable_shards[0].data.shape)


#Visualize Matrix B
jax.debug.visualize_array_sharding(sharded_B)
print ( sharded_B.addressable_shards[0].data.shape)

