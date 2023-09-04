from random import *
import math
import numpy as np
import data_ml2
from matplotlib import pyplot as plt


#définir le nombre de neurones dans la couche cachée et output
nb_neurons_h = 6
nb_neurons_o = data_ml2.targets.shape[1]

#générer des valeurs aléatoires pour chaque poids de biais
rng = np.random.default_rng()

weights_h = -1 + 2*rng.random(size=(nb_neurons_h, data_ml2.input_size))
w_o = -1 + 2*rng.random(size=(nb_neurons_o, nb_neurons_h))

bias_h = -1 + 2*rng.random(size=(nb_neurons_h))
bias_o = -1 + 2*rng.random(size=(nb_neurons_o))




def sigmoid(y):
  return 1 / (1 + math.e**(-y))

def deriveesigmoid(y):
  return math.e**(-y) / (1 + math.e**(-y))**2

def softmax(y):
  exp_y = np.exp(y - np.max(y))
  probs = exp_y / np.sum(exp_y)
  return probs

#categorical cross entropy loss
def categorical_cross_entropy(target, z_o):
  return np.sum(-np.log(z_o + 10**-100)*target)



nb_it = 0
loss_list = []

while 1:
  nb_it += 1
  loss = 0

  d_bias_h = np.zeros_like(bias_h)
  d_bias_o = np.zeros_like(bias_o)

  d_weights_h = np.zeros_like(weights_h)
  d_w_o = np.zeros_like(w_o)

  sample_indices = np.arange(data_ml2.data.shape[0])
  rng.shuffle(sample_indices)    #bagging

  for j in sample_indices:
    #j = randint(0, data_ml2.nb_data-1)           #true stochastic descent
    x = (data_ml2.data[j]).reshape(1, -1)


    target = data_ml2.targets[j]

    y_h = (x @ weights_h.T) + (bias_h)
    z_h = sigmoid(y_h)


    y_o = (z_h @ w_o.T) + (bias_o)
    z_o = softmax(y_o)




    loss += categorical_cross_entropy(target, z_o.squeeze()) / data_ml2.nb_data
    
    d_y_o = (z_o - target)
    d_bias_o = d_y_o.squeeze()
    d_w_o = d_y_o.T @ z_h

    d_y_h = (d_y_o @ w_o)*deriveesigmoid(y_h)
    d_bias_h = d_y_h.squeeze()
    d_weights_h = d_y_h.T @ x

    l_r = 0.75

    bias_h -= l_r*d_bias_h
    bias_o -= l_r*d_bias_o

    weights_h -= l_r*d_weights_h
    w_o -= l_r*d_w_o

  loss_list.append(loss)
  if loss<0.0005 or nb_it>5000:
    break


print("loss",loss,"after",nb_it,"iterations")


x = np.arange(len(loss_list))
plt.plot(x, loss_list)
plt.show()




#partie test
def test():
  try:
    x = np.array([int(q) for q in input("test:  ")]).reshape(1, -1)
  except ValueError:
    pass



  y_h = np.dot(x, weights_h.T) + (bias_h)
  z_h = sigmoid(y_h)

  y_o = np.dot(z_h, w_o.T) + (bias_o)
  z_o = softmax(y_o)



  result = (z_o * 100).squeeze()

  print("paysage", round(result[1], 2), "%")
  print("paysage bas", round(result[2], 2), "%")
  print("pas un paysage", round(result[0], 2), "%")

while 1:
  test()
