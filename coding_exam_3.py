#!/usr/bin/python

############################
# module: coding_exam_3.py
# YOUR NAME
# YOUR A-NUMBER
############################

import cv2
import numpy as np
import argparse
import os
import re
import sys
from sklearn import tree, metrics
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import random
import math
from sklearn.metrics import confusion_matrix

## ============= Problem 1 =================

def eval_dtr_built_from_img_dir(img_dir):
  ## classifier
  classifier = tree.DecisionTreeClassifier(random_state=0)
  data_items = ()
  target
  ## load data from files
  gen_box = ()
  gen_box.append(generate_file_names)
  while gen_box != null:
    gen_box.append(generate_file_names)

  
  ## for each test split value
      ## run_train_test_split
      ## run confusion matrix
      ## 
  for t_val in xrange(0,x,10):
    run_train_test_split(classifier, 1, t_val)
  pass

def generate_file_names(rootdir):
  for path, dirlist, filelist in os.walk(rootdir):
    for file_name in filelist:
      if file_name.find('A'):
        target = 0
      elif file_name.find('B'):
        target = 1
      elif file_name.find('V'):
        target = 2
      elif file_name.find('G'):
        target = 3
      elif file_name.find('D'):
        target = 4
       
      yield (os.path.join(path, file_name),target)
  pass

def run_train_test_split(classifier, n, test_size):
    for i in xrange(n):
        train_data, test_data, train_target, test_target = \
                    train_test_split(data_items, target,
                                     test_size=test_size, random_state=randint(0, 1000))
        dt = classifier.fit(train_data, train_target)
        print 'train/test run ',i,': accuracy = ',(sum(dt.predict(test_data) == test_target)/float(len(test_target)))
        print '------------------------------------------------------'
    return dt

def compute_train_test_confusion_matrix(classifier, test_size):
    train_data, test_data, train_target, test_target = \
                    train_test_split(data_items, target,
                                     test_size=test_size, random_state=randint(0, 1000))
    dt = classifier.fit(train_data, train_target)
    test_predict = dt.predict(test_data)
    cm = confusion_matrix(test_target, test_predict)
    plot_confusion_matrix(cm, ['0','1','2','3','4','5','6','7','8','9'], 'Digits Decision Tree')
    
  
## ================= Problem 2 ==================
    
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
  
def plot_kmeans_clustered_data(temp_file_path, nc):
  ## read file
    ## assuming some way to get a file to be
    ## 
  ## parse hours and temp from file
  ## run kmeans
  ## plot kmeans
  pass
  
## =========== Problem 3 ==================

def displayDependentVars(probDiff=0.1):
  ## your code
  pass

def displayIndependentVars(probDiff=0.1):
  ## your code
  pass

## ============== Problem 4 =====================

def VG():
  ## useless fucntion
  pass

def VG(n):
  if n == 0:
    return 1
  return (((1/(n+1))*(4*n-2))*VG(n-1))

def sum_of_first_n_odd_vms(n):
  sum_vg = 0
  for x in xrange(n+1):
    sum_vg += VG(x)
  return sum_vg
  pass

## for your unit tests
if __name__ == '__main__':
    print sum_of_first_n_odd_vms(2)
    print sum_of_first_n_odd_vms(3)
    
    pass
