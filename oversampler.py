
import copy
import pandas as pd
import numpy as np

class OverSampler():
  def __init__(self, pos_train, neg_train):
    self.pos_train = pos_train
    self.neg_train = neg_train
  
  def get_scale(self, p):
    scale = len(self.pos_train)
    scale /= (len(self.pos_train) + len(self.neg_train))
    if p != 0:
      scale /= p
    scale -= 1
    return scale
  
  def get_over_sample(self, p=0.165):
    scale = self.get_scale(p)
    while scale > 1:
      self.neg_train = pd.concat([self.neg_train, self.neg_train])
      scale -=1
    self.neg_train = pd.concat([self.neg_train, self.neg_train[:int(scale * len(self.neg_train))]])
    print(len(self.pos_train) / (len(self.pos_train) + len(self.neg_train)))
    X_train = pd.concat([self.pos_train, self.neg_train])
    y_train = (np.zeros(len(self.pos_train)) + 1).tolist() + np.zeros(len(self.neg_train)).tolist()
    return (X_train, y_train)
    