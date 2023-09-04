from random import *
import math
import data_ml


wk = []
wl = []
wm = []
wn = []
wo = []

nb_w = 4 #nombre de poids qui entrent dans chaque neurone
#générer des valeurs aléatoires pour chaque poids de chaque neurone
for i in range(nb_w):
  w = -1+2*random()
  wk.append(w)
  
  w = -1+2*random()
  wl.append(w)
  
  w = -1+2*random()
  wm.append(w)
  
  w = -1+2*random()
  wn.append(w)
  
  w = -1+2*random()
  wo.append(w)
  

#générer des valeurs aléatoires pour chaque biais
biask = -1+2*random()
biasl = -1+2*random()
biasm = -1+2*random()
biasn = -1+2*random()
biaso = -1+2*random()


print("wk", wk,"biask",biask)
print("wl", wl,"biasl",biasl)
print("wm", wm,"biasm",biasm)
print("wn", wn,"biasn",biasn)
print("wo", wo,"biaso",biaso)



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
  d_wo = [0,0,0,0]
  d_wk = [0,0,0,0]
  d_wl = [0,0,0,0]
  d_wm = [0,0,0,0]
  d_wn = [0,0,0,0]
  
  
  for j in range(data_ml.nb_data):
    x = data_ml.data[j]
    target = x[4]

    yk = biask
    yl = biasl
    ym = biasm
    yn = biasn

    for i in range(nb_w):
      yk += x[i]*wk[i]
      yl += x[i]*wl[i]
      ym += x[i]*wm[i]
      yn += x[i]*wn[i]
      
    zk = sigmoid(yk)
    zl = sigmoid(yl)
    zm = sigmoid(ym)
    zn = sigmoid(yn)


    yo = biaso
    yo += zk*wo[0]
    yo += zl*wo[1]
    yo += zm*wo[2]
    yo += zn*wo[3]

    zo = sigmoid(yo)


    
        
    loss += (zo - target)**2 / (data_ml.nb_data * 2)
  

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

      

    l_r = 0.1

    biask -= l_r*d_biask
    biasl -= l_r*d_biasl
    biasm -= l_r*d_biasm
    biasn -= l_r*d_biasn
    biaso -= l_r*d_biaso

    for i in range(4):
      wk[i] -= l_r*d_wk[i]
      wl[i] -= l_r*d_wl[i]
      wm[i] -= l_r*d_wm[i]
      wn[i] -= l_r*d_wn[i]
      wo[i] -= l_r*d_wo[i]
  

  if loss<0.0005 or nb_it>10000:
    break

  
print("loss",loss,"after",nb_it,"iterations")

#partie test

def test():
  try:
    x = [int(q) for q in input("test:  ")]
  except ValueError:
    pass
  yk = biask
  yl = biasl
  ym = biasm
  yn = biasn

  for i in range(nb_w):
      yk += x[i]*wk[i]
      yl += x[i]*wl[i]
      ym += x[i]*wm[i]
      yn += x[i]*wn[i]
      
  zk = sigmoid(yk)
  zl = sigmoid(yl)
  zm = sigmoid(ym)
  zn = sigmoid(yn)
    


  #layer 2 
  #output de la deuxième (dernière) couche
  yo = biaso
  yo += zk*wo[0]
  yo += zl*wo[1]
  yo += zm*wo[2]
  yo += zn*wo[3]

  zo = sigmoid(yo)

  result = zo * 100
  print("paysage: ", round(result,2), "%")
  print("pas un paysage: ", round(100-result,2), "%")

while 1:
  test()

