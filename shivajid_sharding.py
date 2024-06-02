import jax
import numpy as np
from  datetime import datetime

#A = jax.numpy.ones([65536, 65536])
#B = jax.numpy.ones([65536, 65536])
def find_duration(dt):
    return (datetime.now() - dt)


dt1 = datetime.now()
print( dt1)
A = np.ones((65536,655362))

print("duration0",find_duration(dt1))
dt1 = datetime.now()
B = np.ones((65536,65536))

print("duration1",find_duration(dt1))

mesh =  jax.sharding.Mesh(np.reshape(jax.devices(), "data"))
sharding =  jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("data"))


dt1 = datetime.now()
sharded_A = jax.device_put(A, sharding)


print("duration2",find_duration(dt1))
dt1 = datetime.now()

sharded_B = jax.device_put(B, sharding)


print("duration3",find_duration(dt1))

dt1 = datetime.now()

jax.debug.visualize_array_sharding(sharded_A)
jax.debug.visualize_array_sharding(sharded_B)


dt1 = datetime.now()

#def add_fn(sharaded_A, sharder_B):
#    return sharded_A + sharded_B

#jit_add_fn = jax.jit(add_fn)

#C = jit_add_fn(sharded_A, sharded_B)

C = jit_add_fn(sharded_A, sharded_B)

print ( C.addressable_shards[0].data.shape)
jax.debug.visualize_array_sharding(C)


print("duration4",find_duration(dt1))

dt1 = datetime.now()

D = C @ sharded_B


print("duration5",find_duration(dt1))
jax.debug.visualize_array_sharding(D)

