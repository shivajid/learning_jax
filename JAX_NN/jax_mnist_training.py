#Thiese are sample code
#
#

import jax.numpy as jnp
import jax
import tensorflow as tf
import tensorflow_datasets as tfds
from jax.scipy.special import logsumexp
import time


print("Environment Check")
print("=="*20)

print(f"JAX backend: {jax.lib.xla_bridge.default_backend()}")
print(f"devices: {jax.devices()}")
print("=="*20)

#Initialize Variables
MATRIX_DIM = 1024
key1 = jax.random.key(0)
key2, key3 = jax.random.split(key1,2)
step_size = 0.001
num_epochs = 100
batch_size =128
IMG_SIZE = 28
IMG_SHAPE = [28,28,1]
LAYERS =[784, 512, 512, 10]

 
W = jax.random.normal(key1, (MATRIX_DIM, MATRIX_DIM), dtype=jax.dtypes.bfloat16) 
B = jax.random.normal(key1, (MATRIX_DIM,1), dtype=jax.dtypes.bfloat16) 
X = jax.random.normal(key2, (MATRIX_DIM, MATRIX_DIM), dtype=jax.dtypes.bfloat16) 

def random_layer_params(m,n,key, scale=1e-2):
 print(f"Matrix dimensions {m} x {n}")
 w_key, b_key = jax.random.split(key)
 #The matrix is transposed
 W_mat = jax.random.normal(w_key, (n,m), dtype=jax.dtypes.bfloat16) # This is transposed, so it in n.m 
 W_mat_scaled = W_mat * scale
 b_mat = jax.random.normal(b_key, (n,), dtype= jax.dtypes.bfloat16)
 b_mat_scaled = b_mat *scale

 return W_mat_scaled, b_mat_scaled

 

def init_network_params(key, sizes):
 keys = jax.random.split(key, len(sizes))
 return [random_layer_params(m, n, k) for m, n, k in zip(sizes[:-1], sizes[1:], keys)]


@jax.jit
def init_NN(W, B, X):
 Y = W@X + B  
 return Y


Y = init_NN(W,B,X)
print("=="*20)
print("Output of init_NN: ")
print(Y)
print(Y.devices())
print("=="*20)


print("=="*20)

layer_transiztions = zip(LAYERS[:-1], LAYERS[1:])
print("=="*20)
print(LAYERS)
print("=="*20)
print(tuple(layer_transiztions))

#Non linear function
@jax.jit
def relu(x):
 return jnp.maximum(0, x)

@jax.jit
def predict(params, image):
 '''
 per image example
 '''
 activations = image

 for W, b in params[:-1]:
  out_layer = W @ activations + b
  activations = relu(out_layer)

 final_W, final_b = params[-1]
 logits = final_W @ activations + final_b
 # normalize with logsum exp
 return logits - logsumexp(logits) 


 
 
params = init_network_params(jax.random.key(0), LAYERS)
#print("=="*20)
#print([(tuple(val)[0]).shape for val in params])
#print("=="*20)

#Image flattened as MNIST has 28 x 28 x 1
random_flattened_image = jax.random.normal(jax.random.key(1), (28 * 28,))

#Forward pass with the image flattened
preds = predict(params, random_flattened_image)

print(preds.shape)


#Lets work with a batch of images
'''
batched_random_flattened_images = jax.random.normal(jax.random.key(0), (10, 28*28))

batched_predict = jax.vmap(predict, in_axes=(None, 0))

batched_preds = batched_predict(params, batched_random_flattened_images)


print(batched_preds.shape)
'''
batched_predict = jax.vmap(predict, in_axes=(None, 0))

def accuracy(params, images, targets):
  target_class = jnp.argmax(targets, axis=1)
  predicted_class = jnp.argmax(batched_predict(params, images), axis=1)
  return jnp.mean(predicted_class == target_class)

def loss(params, images, targets):
  preds = batched_predict(params, images)
  return -jnp.mean(preds * targets)


def update(params, x, y):
  grads = jax.grad(loss)(params, x, y)
  return [(w - step_size * dw, b - step_size * db)
          for (w, b), (dw, db) in zip(params, grads)]

def one_hot(x, k, dtype=jnp.bfloat16):
  """Create a one-hot encoding of x of size k."""
  return jnp.array(x[:, None] == jnp.arange(k), dtype)


'''

@jax.jit
def loss(batched_flattened_images, targets, params):
  batched_predict = jax.vmap(predict, in_axes=(None, 0))
  preds = batched_predict(params, batched_flattened_images)
  return -jnp.mean(preds * targets)

@jax.jit
def update(params, x, y):
  grads = jax.grad(loss)(x,y,params)
  return [(w - step_size * dw, b - step_size * db) for (w, b), (dw, db) in zip(params, grads)]
'''

# Fetch full datasets for evaluation
# tfds.load returns tf.Tensors (or tf.data.Datasets if batch_size != -1)
# You can convert them to NumPy arrays (or iterables of NumPy arrays) with tfds.dataset_as_numpy
mnist_data, info = tfds.load(name="mnist", batch_size=-1,  with_info=True)
mnist_data = tfds.as_numpy(mnist_data)
train_data, test_data = mnist_data['train'], mnist_data['test']
num_labels = info.features['label'].num_classes
h, w, c = info.features['image'].shape
num_pixels = h * w * c
# Full train set
train_images, train_labels = train_data['image'], train_data['label']
train_images = jnp.reshape(train_images, (len(train_images), num_pixels))
train_labels = one_hot(train_labels, num_labels)

# Full test set
test_images, test_labels = test_data['image'], test_data['label']
test_images = jnp.reshape(test_images, (len(test_images), num_pixels))
test_labels = one_hot(test_labels, num_labels)

def get_train_batches():
  ds = tfds.load(name="mnist",split="train", as_supervised=True)
  ds = ds.batch(batch_size).prefetch(10)
  return tfds.as_numpy(ds)

#Start the training loop
for epoch in range(num_epochs):
  start_time = time.time()
  for x,y in get_train_batches():
    x = jnp.reshape(x,(len(x), num_pixels))
    #x = x.astype(jax.dtypes.float0)
    y = one_hot(y, num_labels)
    params = update(params, x,y)
  epoch_time = time.time() - start_time

  train_acc = accuracy(params, train_images, train_labels)
  test_acc = accuracy(params, test_images, test_labels)
  print("Epoch {} in {:0.2f} sec".format(epoch, epoch_time))
  print("Training set accuracy {}".format(train_acc))
  print("Test set accuracy {}".format(test_acc))
  














