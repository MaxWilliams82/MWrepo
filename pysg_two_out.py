from random import *
import math
import numpy as np
import data_ml2


#générer des valeurs aléatoires pour chaque poids de chaque neurone
wk = np.array([-1+2*random(), -1+2*random(), -1+2*random(), -1+2*random()])
wl = np.array([-1+2*random(), -1+2*random(), -1+2*random(), -1+2*random()])
wm = np.array([-1+2*random(), -1+2*random(), -1+2*random(), -1+2*random()])
wn = np.array([-1+2*random(), -1+2*random(), -1+2*random(), -1+2*random()])
wo = np.array([-1+2*random(), -1+2*random(), -1+2*random(), -1+2*random()])
wp = np.array([-1+2*random(), -1+2*random(), -1+2*random(), -1+2*random()])

#générer des valeurs aléatoires pour chaque biais
biask = -1+2*random()
biasl = -1+2*random()
biasm = -1+2*random()
biasn = -1+2*random()
biaso = -1+2*random()
biasp = -1+2*random()

#imprimer les poids et biais
print("wk", wk,"biask",biask)
print("wl", wl,"biasl",biasl)
print("wm", wm,"biasm",biasm)
print("wn", wn,"biasn",biasn)
print("wo", wo,"biaso",biaso)
print("wp", wo,"biasp",biaso)


def sigmoid(y):
  return 1 / (1 + math.e**(-y))

def deriveesigmoid(y):
  return math.e**(-y) / (1 + math.e**(-y))**2



nb_it = 0

while 1:
  nb_it += 1 
  loss = 0
  d_biaso = 0
  d_biask = 0
  d_biasl = 0
  d_biasm = 0
  d_biasn = 0
  d_biasp = 0
  d_wo = [0,0,0,0]
  d_wk = [0,0,0,0]
  d_wl = [0,0,0,0]
  d_wm = [0,0,0,0]
  d_wn = [0,0,0,0]
  d_wp = [0,0,0,0]
  
  
  for j in range(data_ml2.nb_data):
    x = np.array(data_ml2.data[j][0:4])
    #print("data :", x)
    target = data_ml2.data[j][4]
    target2 = data_ml2.data[j][5]
    #print("target :", target)

    yk = np.dot(x, wk) + (biask)
    yl = np.dot(x, wl) + (biasl)
    ym = np.dot(x, wm) + (biasm)
    yn = np.dot(x, wn) + (biasn)

    zk = sigmoid(yk)
    zl = sigmoid(yl)
    zm = sigmoid(ym)
    zn = sigmoid(yn)

    xo = np.array([zk, zl, zm, zn])
    xp = np.array([zk, zl, zm, zn])

    yo = np.dot(xo, wo) + (biaso)
    yp = np.dot(xp, wp) + (biasp)

    zo = sigmoid(yo)
    zp = sigmoid(yp)


        
    loss += (zo - target)**2 / (data_ml2.nb_data * 2)
    loss += (zp - target2)**2 / (data_ml2.nb_data * 2)

    d_biaso += (zo - target)*deriveesigmoid(yo)
    d_wo[0] += (zo - target)*deriveesigmoid(yo)*zk
    d_wo[1] += (zo - target)*deriveesigmoid(yo)*zl
    d_wo[2] += (zo - target)*deriveesigmoid(yo)*zm
    d_wo[3] += (zo - target)*deriveesigmoid(yo)*zn

    d_biask += (zo - target)*deriveesigmoid(yo)*wo[0]*deriveesigmoid(yk)
    d_biasl += (zo - target)*deriveesigmoid(yo)*wo[1]*deriveesigmoid(yl)
    d_biasm += (zo - target)*deriveesigmoid(yo)*wo[2]*deriveesigmoid(ym)
    d_biasn += (zo - target)*deriveesigmoid(yo)*wo[3]*deriveesigmoid(yn)

    for i in range(4):
      d_wk[i] += (zo - target)*deriveesigmoid(yo)*wo[0]*deriveesigmoid(yk)*x[i]
      d_wl[i] += (zo - target)*deriveesigmoid(yo)*wo[1]*deriveesigmoid(yl)*x[i]
      d_wm[i] += (zo - target)*deriveesigmoid(yo)*wo[2]*deriveesigmoid(ym)*x[i]
      d_wn[i] += (zo - target)*deriveesigmoid(yo)*wo[3]*deriveesigmoid(yn)*x[i]
    
    d_biasp += (zp - target)*deriveesigmoid(yp)
    d_wp[0] += (zp - target)*deriveesigmoid(yp)*zk
    d_wp[1] += (zp - target)*deriveesigmoid(yp)*zl
    d_wp[2] += (zp - target)*deriveesigmoid(yp)*zm
    d_wp[3] += (zp - target)*deriveesigmoid(yp)*zn

    d_biask += (zp - target)*deriveesigmoid(yp)*wp[0]*deriveesigmoid(yk)
    d_biasl += (zp - target)*deriveesigmoid(yp)*wp[1]*deriveesigmoid(yl)
    d_biasm += (zp - target)*deriveesigmoid(yp)*wp[2]*deriveesigmoid(ym)
    d_biasn += (zp - target)*deriveesigmoid(yp)*wp[3]*deriveesigmoid(yn)

    for i in range(4):
      d_wk[i] += (zp - target)*deriveesigmoid(yp)*wp[0]*deriveesigmoid(yk)*x[i]
      d_wl[i] += (zp - target)*deriveesigmoid(yp)*wp[1]*deriveesigmoid(yl)*x[i]
      d_wm[i] += (zp - target)*deriveesigmoid(yp)*wp[2]*deriveesigmoid(ym)*x[i]
      d_wn[i] += (zp - target)*deriveesigmoid(yp)*wp[3]*deriveesigmoid(yn)*x[i]
      

    l_r = 0.1

    biask -= l_r*d_biask
    biasl -= l_r*d_biasl
    biasm -= l_r*d_biasm
    biasn -= l_r*d_biasn
    biaso -= l_r*d_biaso
    biasp -= l_r*d_biasp
    
    for i in range(4):
      wk[i] -= l_r*d_wk[i]
      wl[i] -= l_r*d_wl[i]
      wm[i] -= l_r*d_wm[i]
      wn[i] -= l_r*d_wn[i]
      wo[i] -= l_r*d_wo[i]
      wp[i] -= l_r*d_wp[i]

  

  if loss<0.0005 or nb_it>10000:
    break

print("loss",loss,"after",nb_it,"iterations")


#partie test
while 1:
  try:
    x = [int(q) for q in input("test:  ")]
  except ValueError:
    break

  yk = np.dot(x, wk) + (biask)
  yl = np.dot(x, wl) + (biasl)
  ym = np.dot(x, wm) + (biasm)
  yn = np.dot(x, wn) + (biasn)

  zk = sigmoid(yk)
  zl = sigmoid(yl)
  zm = sigmoid(ym)
  zn = sigmoid(yn)

  xo = np.array([zk, zl, zm, zn])
  xp = np.array([zk, zl, zm, zn])

  yo = np.dot(xo, wo) + (biaso)
  yp = np.dot(xp, wp) + (biasp)

  zo = sigmoid(yo)
  zp = sigmoid(yp)


  result = zo * 100
  print("top: ", round(result,2), "%")
  print("no horizontal line: ", round(100-result,2), "%")

  result2 = zp * 100
  print("bottom: ", round(result2,2), "%")
  print("no horizontal line: ", round(100-result2,2), "%")