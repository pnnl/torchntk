import tensorflow as tf
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
#import logging
#logging.getLogger('tensorflow').disabled = True

mpl.rcParams['figure.figsize'] = (8, 6)
#tf.random.set_seed(1234)

def computing_Hessian(m=10, n_0=10, n_1=10, n_2=10, 
                      activation=tf.nn.relu, title='relu', xlim=[-8,8], bins = np.linspace(-8, 8, 120)):
  x = tf.random.normal([m, n_0])
  layer1 = tf.keras.layers.Dense(n_1, activation=activation)
  layer2 = tf.keras.layers.Dense(n_2, activation=tf.keras.activations.linear)
  with tf.GradientTape(persistent=True) as t2:
    with tf.GradientTape(persistent=True) as t1:
      x1 = layer1(x)
      x2 = layer2(x1)
    g1,g2 = t1.jacobian(x2, [layer1.kernel,layer2.kernel])
  f1,f2 = t2.jacobian(g1, [layer1.kernel,layer2.kernel])
  print(g1.shape)
  print(g2.shape)
  print(f1.shape)
  print(f2.shape)
  n_params1 = tf.reduce_prod(layer1.kernel.shape)
  n_params2 = tf.reduce_prod(layer2.kernel.shape)
  n_input = tf.reduce_prod(x2.shape)
  g1_vec = tf.reshape(g1, [n_input,n_params1])
  print(g1_vec.shape)
  g2_vec = tf.reshape(g2, [n_input,n_params2])
  print(g2_vec.shape)
  g = tf.concat([g1_vec, g2_vec], 1)
  print(g.shape)
  H_0 = tf.linalg.matmul(tf.transpose(g),g)/m 
  print(H_0.shape) 
  f1_vec = tf.reshape(f1, [m,n_2,n_params1,n_params1])
  print(f1_vec.shape)
  f2_vec = tf.reshape(f2, [m,n_2,n_params1,n_params2])
  print(f2_vec.shape)
  f11 = tf.concat([f1_vec, f2_vec], -1)
  print(f11.shape)
  f3_vec = tf.transpose(f2_vec,perm=[0,1,3,2])
  print(f3_vec.shape)
  f4_vec = tf.zeros([m,n_2,n_params2,n_params2])
  print(f4_vec.shape)
  f22 = tf.concat([f3_vec, f4_vec], -1)
  print(f22.shape)
  H_11 = tf.concat([f11, f22], 2)
  print(H_11.shape)
  H_1=tf.zeros([H_0.shape[0],H_0.shape[1]])
  for i in range(m):
    for j in range(n_2):
      H_1 = H_1 + x2[i,j]*H_11[i,j,:,:]
  H_1 = H_1/m
  print(H_1.shape)
  H = H_0+H_1
  eigs0,_ = tf.linalg.eigh(H)
  initializer = tf.keras.initializers.Orthogonal()
  O = initializer(shape=H_1.shape)
  H_new = H_0 + tf.linalg.matmul(O,tf.linalg.matmul(H_1,tf.transpose(O)))
  eigs1,_ = tf.linalg.eigh(H_new)
  plt.hist(eigs0.numpy(), bins, alpha=0.5, label='original')
  #plt.hist(eigs1.numpy(), bins, alpha=0.5, label='new')
  plt.legend(loc='upper right')
  plt.title(title)
  #plt.xlim(xlim)
  plt.show()
  plt.savefig("Hessian_m=%d_n0=%d_n_1=%d.png"% (m,n_0,n_1))
  return eigs0.numpy(), eigs1.numpy()

eig_old,eig_new=computing_Hessian(m=140, n_0=13, n_1=13, n_2=2,
                      activation=tf.nn.sigmoid, title='sigmoid', xlim=[-2,2], bins = 45)
#eig_old,eig_new=computing_Hessian(m=150, n_0=12, n_1=12, n_2=2, activation=tf.nn.relu, title='relu', xlim=[-2,2], bins = 50)
print(eig_old, flush=True)
print(eig_new, flush=True)



