#modifications version :
#-intégration arrays numpy







from random import *
import math
import numpy as np
import data_ml
from matplotlib import pyplot as plt


#générer des valeurs aléatoires pour chaque poids de biais
rng = np.random.default_rng()

weights_h = -1 + 2*rng.random(size=(4,4))
w_o = -1 + 2*rng.random(size=(1,4))

bias_h = -1 + 2*rng.random(size=(4))
bias_o = -1 + 2*rng.random(size=(1))



def sigmoid(y):
  return 1 / (1 + math.e**(-y))

def deriveesigmoid(y):
  return math.e**(-y) / (1 + math.e**(-y))**2






nb_it = 0
loss_list = []

while 1:
  nb_it += 1
  loss = 0

  d_bias_h = np.zeros(shape=(4))
  d_bias_o = np.zeros(shape=(1))

  d_weights_h = np.zeros(shape=(4, 4))
  d_w_o = np.zeros(shape=(1, 4))


  for j in range(data_ml.nb_data):
    x = np.array(data_ml.data[j][0:4])
    #print("data :", x)
    target = data_ml.data[j][4]
    #print("target :", target)

    y_h = np.dot(x, weights_h.T) + (bias_h)
    z_h = sigmoid(y_h)

    y_o = np.dot(z_h, w_o.T) + (bias_o)
    z_o = sigmoid(y_o)



    #mean squared error
    loss += (z_o - target)**2 / (data_ml.nb_data * 2)


    d_bias_o += (z_o - target)*deriveesigmoid(y_o)
    d_w_o += (z_o - target)*deriveesigmoid(y_o)*z_h

    d_y = (z_o - target)*deriveesigmoid(y_o)*w_o*deriveesigmoid(y_h)
    d_bias_h += d_y.squeeze()
    d_weights_h += d_y.T @ x.reshape(1, -1)

    l_r = 0.1

    bias_h -= l_r*d_bias_h
    bias_o -= l_r*d_bias_o

    weights_h -= l_r*d_weights_h
    w_o -= l_r*d_w_o

  loss_list.append(loss)
  if loss<0.0005 or nb_it>1000:
    break


print("loss",loss,"after",nb_it,"iterations")


x = np.arange(len(loss_list))
plt.plot(x, loss_list)
plt.show()


#partie test
def test():
  try:
    x = [int(q) for q in input("test:  ")]
  except ValueError:
    pass



  y_h = np.dot(x, weights_h.T) + (bias_h)
  z_h = sigmoid(y_h)

  y_o = np.dot(z_h, w_o.T) + (bias_o)
  z_o = sigmoid(y_o)



  result = z_o * 100
  print("paysage: ", result, "%")
  print("pas un paysage: ", 100-result, "%")

while 1:
  test()
