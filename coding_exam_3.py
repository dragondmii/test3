#!/usr/bin/python

############################
# module: coding_exam_3.py
# Robert Epstein
# A01092594
############################
from __future__ import division
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
  ##
  ## I CANT MAKE THIS WORK SORRY MY CODE DOESNT COMPILE
  ##
  gen_box = ()
  gen_box.append(generate_file_names)
  while gen_box != null:
    gen_box.append(generate_file_names)

  
  ## for each test split value
      ## run confusion matrix
      ## 
  for t_val in xrange(0,x,10):
    cm = Compute_train_test_confusion_matrix(classifier, t_val)
    print 'Test size = "', t_val
    print '\n'
    print 'Classification report:\n'
    ## I dont know how to build that report
    print 'Confusion matrix:\n'
    print cm
  
  pass

## this is an attempt to build the data into data_items and training but I dont
##   think I have the data types or format right
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

def compute_train_test_confusion_matrix(classifier, test_size):
    train_data, test_data, train_target, test_target = \
                    train_test_split(data_items, target,
                                     test_size=test_size, random_state=randint(0, 1000))
    dt = classifier.fit(train_data, train_target)
    test_predict = dt.predict(test_data)
    cm = confusion_matrix(test_target, test_predict)
    
  
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

ageGroups = (20,30,40,50,60,70)
peopleInAgeGroup= {20:326,30:335,40:351,50:323, 60:327, 70:338}
accidentsInAgeGroup = {20:169,30:94,40:136,50:63,60:214,70:229}
numOfPeople= 2000
numOfAccidents = 905



def displayDependentVars(probDiff=0.1):
  ## your code
  for x in ageGroups:
    diffProbs = abs(condProbOfPurchaseGivenAgeGroup(x) - probOfPurchase())
    if (diffProbs > probDiff):
      print "TA and AG=",x," are Dependent"

  pass

def displayIndependentVars(probDiff=0.1):
  for x in ageGroups:
      diffProbs = abs(condProbOfPurchaseGivenAgeGroup(x) - probOfPurchase())
      if (diffProbs < probDiff):
          print "TA and AG=",x," are independent"
          
  pass

## P(Purchase | AgeGroup = x)
def probOfPurchaseGivenAgeGroup(x):
    return float(accidentsInAgeGroup[x])/peopleInAgeGroup[x]

## P(AgeGroup=x)
def probOfAgeGroup(x):
    return float(peopleInAgeGroup[x])/numOfPeople

## P(Purchase) = prob of buying something
def probOfPurchase():
    return float(numOfAccidents)/numOfPeople
 
## P(Purchase, AG=x) 
def probOfPurchaseAndAgeGroup(x):
    return float(accidentsInAgeGroup[x])/numOfPeople

## P(Purchase | AgeGroup = x) = P(Purchase, AgeGroup=x)/P(AgeGroup=x)
def condProbOfPurchaseGivenAgeGroup(x):
    return probOfPurchaseAndAgeGroup(x)/probOfAgeGroup(x)


## ============== Problem 4 =====================

def VG(n):
  if n == 0:
    return 1
  x = float(1/(n+1))
  y = (4*n-2)
  z = x*y
  return (z*VG(n-1))

def sum_of_first_n_odd_vms(n):
  sum_vg = 0
  for x in xrange(n+1):
    if x%2 != 0:
      sum_vg += VG(x)
  return sum_vg
  pass

## for your unit tests
if __name__ == '__main__':
  displayDependentVars()   
  pass
