import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from tensorflow import keras
from tensorflow.keras import initializers
import pickle
import os

font = {'size': 15}
matplotlib.rc('font', **font)
np.random.seed(321)
tf.random.set_seed(456)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--steps', type=int, default=5000)
parser.add_argument('--batch', type=int, default=32)
parser.add_argument('--n', type=int, default=8000)
parser.add_argument('--eta', type=float, default=0.001)
#parser.add_argument('--lda', type=float, default=0.2)
parser.add_argument('--gamma1', type=float, default=0.1)
parser.add_argument('--gamma2', type=float, default=1.0)
args = parser.parse_args()

gamma1 = args.gamma1 #gamma1 = d/n
gamma2 = args.gamma2 # gamma2 = h/n
batch = args.batch
#lda = args.lda
# Learning rate
learning_rate = args.eta
sample_dim = args.n
# Number of steps
epoch = args.steps

# Setting parameters
test_sample_dim = args.n
input_dim = int(sample_dim*gamma1)
weight_dim = int(sample_dim*gamma2)
weight_dim_true = 10
t_n = 1.0 / np.sqrt(weight_dim)
s_n = 1.0 / np.sqrt(input_dim)
L = 3

# Define the nonlinear activation function
c = np.sqrt(0.21747 / (2 * np.sqrt(2 * np.pi)))


def activ_function(x):
    return (keras.backend.sigmoid(x) - 0.5) / c


b = 2 * (0.258961) / np.sqrt(2 * np.pi) / 0.208276
a = 2 * 0.0561939 / (np.sqrt(2 * np.pi) * (c ** 2))


def dactiv_function(x):
    return np.exp(-x) / (((1 + np.exp(-x)) ** 2) * c)

# Generate the training data
coeffi_1 = np.random.normal(size=(weight_dim_true, input_dim))
coeffi_2 = np.random.normal(size=(1, weight_dim_true))


def f_aim(x):
    x_sum = np.inner(coeffi_2,activ_function(np.inner(coeffi_1, x)))
    return tf.math.sign(x_sum)


train_data = np.random.normal(size=(sample_dim, input_dim))*s_n
train_labels = np.array(list(map(f_aim, train_data)))
#train_data_linear = train_data*np.sqrt(2)*b
# Generate the test data
test_data = np.random.normal(size=(test_sample_dim, input_dim))*s_n
test_labels = np.array(list(map(f_aim, test_data)))
#test_data_linear = test_data*np.sqrt(2)*b

def get_dataset():
    batch_size = batch
    x_train = train_data
    y_train = train_labels
    x_test = test_data
    y_test = test_labels

    return (
        tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size),
        tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size),
    )

# initialize tf.distribute.MirroredStrategy
strategy = tf.distribute.MirroredStrategy()
print("Number of devices: {}".format(strategy.num_replicas_in_sync), flush=True)

# Create the model
with strategy.scope():
  inputs = tf.keras.Input(shape=(input_dim,))
  x_10 = tf.keras.layers.Dense(weight_dim, activation=activ_function,
                             use_bias=False, kernel_initializer= initializers.RandomNormal(mean=0., stddev=1.))(inputs)
  x_1 = keras.layers.Lambda(lambda x: x * t_n)(x_10)
  layer_1 = keras.models.Model(inputs, x_1, name="first_layer")
  x_20 = tf.keras.layers.Dense(weight_dim, activation=activ_function,
                             use_bias=False, kernel_initializer= initializers.RandomNormal(mean=0., stddev=1.))(x_1)
  x_2 = keras.layers.Lambda(lambda x: x * t_n)(x_20)
  layer_2 = keras.models.Model(inputs, x_2, name="second_layer")
  x_30 = tf.keras.layers.Dense(weight_dim, activation=activ_function,
                             use_bias=False, kernel_initializer= initializers.RandomNormal(mean=0., stddev=1.))(x_2)
  x_3 = keras.layers.Lambda(lambda x: x * t_n)(x_30)
  layer_3 = keras.models.Model(inputs, x_3, name="third_layer")
  outputs = tf.keras.layers.Dense(1, use_bias=False, kernel_initializer= initializers.RandomNormal(mean=0., stddev=1.))(x_3)
  model = tf.keras.Model(inputs=inputs, outputs=outputs)
  #model.summary(line_length=None, positions=None, print_fn=None)
  optimizer = tf.keras.optimizers.Adam(learning_rate)
  model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['accuracy'])
# Records the weights throughout the training process
X_0 = train_data
Y_0 = train_labels
X_1 = []
X_2 = []
X_3 = []

X_1.append(layer_1.predict(train_data))
X_2.append(layer_2.predict(train_data))
X_3.append(layer_3.predict(train_data))
Ws = []


class MyCallback(keras.callbacks.Callback):
    def __init__(self):
        self.Y_pred = []

    def on_epoch_begin(self, epoch, logs={}):
        X_1.append(layer_1.predict(train_data))
        X_2.append(layer_2.predict(train_data))
        X_3.append(layer_3.predict(train_data))
        self.Y_pred.append(model.predict(test_data))
        Ws.append(model.get_weights())


callback = MyCallback()
train_dataset, test_dataset = get_dataset()
history = model.fit(train_dataset, epochs=epoch, verbose=False, validation_data=test_dataset,callbacks=[callback])

regr = LinearRegression()
regr.fit(train_data, train_labels)
train_y_pred = regr.predict(train_data)
test_y_pred = regr.predict(test_data)
train_loss_reg = mean_squared_error(train_labels, train_y_pred)
test_loss_reg = mean_squared_error(test_labels, test_y_pred)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.axhline(y=train_loss_reg,color='black',linestyle='--')
plt.axhline(y=test_loss_reg,color='black',linestyle='--')
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train_NN', 'test_NN'], loc='upper right')
plt.savefig('./results/Trainig_testing_loss_L=%d_n=%d_d=%d_batch=%d_lr=%g' % (L, sample_dim, weight_dim,batch,learning_rate))
np.save('./results/NNhistory_n=%d_batch=%d_lr=%g.npy' % (sample_dim,batch,learning_rate),history.history)
np.save('./results/history_L=%d.npy' % L,history.history)

ylim = [0, 70]
bins = 50

def cross(X):
    return np.dot(X, np.transpose(X))

# Histogram eigenvalues of each X'X

def histeig(eigs, ax, xgrid=None, dgrid=None, bins=100, xlim=None, ylim=None, title=None):
    if xlim is not None:
        eigs = eigs[np.nonzero(eigs <= xlim[1])[0]]
        h = ax.hist(eigs, bins=np.linspace(xlim[0], xlim[1], num=bins))
    else:
        h = ax.hist(eigs, bins=bins)
    if xgrid is not None:
        space = h[1][1] - h[1][0]
        ax.plot(xgrid, dgrid * len(eigs) * space, 'r', label='Prediction at initial', linewidth=2)
        ax.legend()
    ax.set_title(title, fontsize=12)
    ax.set_xlim(xlim)
    if ylim is None:
        ax.set_ylim([0, max(h[0]) * 1.5])
    else:
        ax.set_ylim(ylim)
    return ax

X = [X_0, X_1, X_2, X_3]
with open('./results/weights_L=%d_n=%d_d=%d_batch=%d_lr=%g'% (L,sample_dim,weight_dim,batch,learning_rate),'wb') as f:
         pickle.dump(Ws,f)
with open('./results/hidden_outputs_L=%d_n=%d_h=%d_batch=%d_lr=%g'% (L,sample_dim,weight_dim,batch,learning_rate),'wb') as f:
         pickle.dump(X,f)
        
epoch_list = list(range(0,epoch,100))

# Plot empirical spectrum of CK matrix at different epochs

eigs_initial = np.linalg.eigvalsh(cross(X[L][0]))
f, ax = plt.subplots(1, 1)
histeig(eigs_initial, ax, xgrid=None, dgrid=None, bins=bins, ylim=ylim, title='Initial CK spectrum')
plt.savefig('./results/initial_CK_L=%d_n=%d_h=%d_batch=%d_lr=%g.png'% (L,sample_dim,weight_dim,batch,learning_rate))
for k in epoch_list:
    eigs = np.linalg.eigvalsh(cross(X[L][k]))
    f, ax = plt.subplots(1, 1)
    histeig(eigs, ax, xgrid=None, dgrid=None, bins=bins, ylim=ylim, title='Epoch=%d CK spectrum' % k)
    plt.savefig('./results/Epoch=%d_CK_L=%d_n=%d_h=%d_batch=%d_lr=%g.png'% (k,L,sample_dim,weight_dim,batch,learning_rate))

# Plot empirical spectrum of gram of the weight matrix at different epochs 
    
eigs_initial_w = np.linalg.eigvalsh(cross(t_n*Ws[0][0]))
f, ax = plt.subplots(1, 1)
histeig(eigs_initial_w, ax, xgrid=None, dgrid=None, bins=bins, ylim=ylim,title='Initial first-layer weight')
plt.savefig('./results/initial_weight_L=%d_n=%d_h=%d_batch=%d_lr=%g.png'% (L,sample_dim,weight_dim,batch,learning_rate))
for k in range(0,60,3):
    eigs = np.linalg.eigvalsh(cross(t_n*Ws[k][0]))
    f, ax = plt.subplots(1, 1)
    histeig(eigs, ax, xgrid=None, dgrid=None, bins=bins, ylim=ylim, xlim=None, title='Epoch=%d first-layer weight' % k)
    plt.savefig('./results/Epoch=%d_weight_L=%d_n=%d_h=%d_batch=%d_lr=%g.png'% (k,L,sample_dim,weight_dim,batch,learning_rate))

# Save empirical NTKs and their eigenvalue spectra
def compute_NTK(step):
  Ds = [[]]
  Ds.append(dactiv_function(np.dot(np.transpose(Ws[step][0]),np.transpose(X[0]))))
  for l in range(1,L):
      Ds.append(dactiv_function(np.dot(np.transpose(Ws[step][l]),np.transpose(X[l][step]))))
  KNTK_final = cross(X[L][step])
  for l in range(1,L+1):
      if l==1:
          XtX = cross(X[0])
      else:
          XtX = cross(X[l-1][step])
      S = np.zeros((weight_dim,sample_dim))
      for i in range(sample_dim):
          s =Ws[step][L]*t_n
          for k in range(L,l-1,-1):
              s = Ds[k][:,i].reshape((weight_dim,1))*s
              if k > l:
                  s = np.dot(Ws[step][k-1],s)*t_n
          S[:,i] = np.transpose(s)
      KNTK_final += cross(np.transpose(S)) * XtX
  eigs_NTK_final = np.linalg.eigvalsh(KNTK_final)
  with open('./results/NTK_final_L=%d_n=%d_d=%d_epoch=%d_batch=%d_lr=%g'% (L,sample_dim,weight_dim,step,batch,learning_rate),'wb') as f:
         pickle.dump(KNTK_final,f)
  return eigs_NTK_final

for k in epoch_list:
    eigs = compute_NTK(k)
    f, ax = plt.subplots(1, 1)
    histeig(eigs, ax, xgrid=None, dgrid=None, bins=bins, ylim=ylim, title='Epoch=%d NTK spectrum' % k)
    plt.savefig('./results/Epoch=%d_L=%d_NTK_n=%d_h=%d_batch=%d_lr=%g.png'% (k,L,sample_dim,weight_dim,batch,learning_rate))