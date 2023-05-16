import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import sys

from ML_functions import *

#python .py ML_Methods state SR_repeat SR_0

#ML methods - ANN, KNN, SR
#states - s,l,v,e,d
#SR_repeat - how many times to repeat the SR algorithm
#SR_0 - 1 - use D0 and eta0, 0 - don't use


type=0

# States/Types on which to train the ML algorithm. s - supercritical, l - liquid, v - vapour, e - viscosity, d - viscosity and liquid diffusion
# The s,l and v values can be submitted in any combination and any order. e and d have to be on their own.

state=""
if len(sys.argv)>2:state=sys.argv[2]
while len(state)<1: state=input("What state? ")

#gets ML data
X,y,X_scaler,y_scaler,state,y_out=get_ML_data(state)

#splits the data for training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2020)

if state!='d':
    y_test=np.exp(np.ravel(y_scaler.inverse_transform(y_test.reshape(1,-1))))
else:
    y_test=np.exp(y_scaler.inverse_transform(y_test))

if len(y_out)>1:y_out=["Dual"]
Suffix=f"{y_out[0].split('_')[0]}_out"

Suffix+=f"_{state}"

SR_inputs=[]
if len(sys.argv)>3:SR_inputs.append(sys.argv[3])
if len(sys.argv)>4:SR_inputs.append(sys.argv[4])

ML_type=""
if len(sys.argv)>1:ML_type=sys.argv[1]
while len(ML_type)<1: ML_type=input("Which ML method? ")
if "ANN" in ML_type: do_ANN(X_train, X_test, y_train, y_test, X_scaler, y_scaler, Suffix)
if "KNN" in ML_type: do_KNN(X_train, X_test, y_train, y_test, X_scaler, y_scaler, Suffix)
if "SR" in ML_type: do_SR(X_train, X_test, y_train, y_test, X_scaler, y_scaler, Suffix, input_args=SR_inputs)
