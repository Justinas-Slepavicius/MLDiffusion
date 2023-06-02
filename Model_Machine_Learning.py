import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import sys

from ML_functions import *

#python .py ML_Methods state SR_repeat CE

#ML methods - ANN, KNN, SR
#states - s,l,v,e,d
#SR_repeat - how many times to repeat the SR algorithm
#CE - 1 - use D0 and eta0, 0 - don't use


type=0

# States/Types on which to train the ML algorithm. s - supercritical, l - liquid, v - vapour, e - viscosity, d - viscosity and liquid diffusion
# The s,l and v values can be submitted in any combination and any order. e has to be separate.

state=""
if len(sys.argv)>2:state=sys.argv[2]
while len(state)<1: state=input("What state? ")

SR_inputs=[]
if len(sys.argv)>3:SR_inputs.append(sys.argv[3])
if len(sys.argv)>4:SR_inputs.append(int(sys.argv[4]))
with_CE=SR_inputs[1]

#gets ML data
X,y,X_scaler,y_scaler,state,y_out=get_ML_data(state,with_CE)

#splits the data for training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2020)

y_test=np.ravel(y_scaler.inverse_transform(y_test.reshape(1,-1)))
if not with_CE:
    y_test=np.exp(y_test)


if with_CE:
    X_test_scaled=X_scaler.inverse_transform(X_test)
    for i in range(len(y_test)):
        CE=calc_CE(X_test_scaled[i][0],X_test_scaled[i][1],state)
        y_test[i]*=CE
        

Suffix=f"{y_out[0].split('_')[0]}_out"

Suffix+=f"_{state}"

if with_CE:
    if state=='e':
        Suffix+="_eta0"
    else:
        Suffix+="_D0"

ML_type=""
if len(sys.argv)>1:ML_type=sys.argv[1]
while len(ML_type)<1: ML_type=input("Which ML method? ")
if "ANN" in ML_type: do_ANN(X_train, X_test, y_train, y_test, X_scaler, y_scaler, with_CE, Suffix)
if "KNN" in ML_type: do_KNN(X_train, X_test, y_train, y_test, X_scaler, y_scaler, with_CE, Suffix)
if "SR" in ML_type: do_SR(X_train, X_test, y_train, y_test, X_scaler, y_scaler, Suffix, input_args=SR_inputs)
