import jax
import datetime
import sys
import numpy as np


MATRIX_DIM = 32768
#MATRIX_DIM = MATRIX_DIM * 2
STEPS = 100

A = jax.numpy.ones([MATRIX_DIM, MATRIX_DIM], dtype="bfloat16")
B = jax.numpy.ones([MATRIX_DIM, MATRIX_DIM], dtype="bfloat16")
a_np = np.ones((MATRIX_DIM, MATRIX_DIM), dtype='float32')
b_np = np.ones((MATRIX_DIM, MATRIX_DIM), dtype='float32')
c_np = a_np + b_np

num_bytes = A.size * 4 #Since it is fp32
total_num_bytes_cross_hbm = num_bytes * 3 # 3 copies, A, B and operation on A, B

total_flops = MATRIX_DIM * MATRIX_DIM
#jax.profiler.start_trace("/tmp/profile_me")

print (sys.getsizeof(A))
print(num_bytes)
print(jax.devices())
print(A.devices())
print(B.devices())
print(sys.getsizeof(a_np)/(1024*1024*1024), " GB")
print(sys.getsizeof(c_np)/(1024*1024*1024), " GB")

#A_sh = jax.numpy.ones([1024, 1024])

mesh = jax.sharding.Mesh(jax.devices(), "myaxis")
sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("myaxis"))
sharded_A = jax.device_put(A, mesh)
sharded_B = jax.device_put(B, mesh)



def f(A,B):
  for i in range(100):
    C = A+B
  return C

jit_f =  jax.jit(f)

start_time= datetime.datetime.now()

val =  jit_f(A,B)

end_time = datetime.datetime.now() 

val2 = jit_f(sharded_A, sharded_B)

print(val2)

#jax.profiler.stop_trace()
time_per_step = (end_time - start_time).total_seconds()/STEPS

print ("average time per step", time_per_step)

print ("flops per time per step", (total_flops/time_per_step)/10**12)   

print ("Total bytes crossing hbm per second", total_num_bytes_cross_hbm/time_per_step/10**9)
