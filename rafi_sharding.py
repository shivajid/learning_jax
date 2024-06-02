import jax
import datetime


MATRIX_DIM = 32768
STEPS = 100


A = jax.numpy.ones([MATRIX_DIM, MATRIX_DIM])
B = jax.numpy.ones([MATRIX_DIM, MATRIX_DIM])

num_bytes = A.size * 4 #Since it is fp32
total_num_bytes_cross_hbm = num_bytes * 3 # 3 copies, A, B and operation on A, B

total_flops = MATRIX_DIM * MATRIX_DIM
jax.profiler.start_trace("/tmp/profile_me")


start_time= datetime.datetime.now()

for i in range(100):
    C = A+B


end_time = datetime.datetime.now() 

jax.profiler.stop_trace()
time_per_step = (end_time - start_time).total_seconds()/STEPS

print ("average time per step", time_per_step)

print ("flops per time per step", (total_flops/time_per_step)/10**12)   

print ("Total bytes crossing hbm per second", total_num_bytes_cross_hbm/time_per_step/10**9)
