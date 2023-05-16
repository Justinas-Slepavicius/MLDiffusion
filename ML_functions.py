import numpy as np
import pandas as pd
import math
from random import randint
import sympy
from sklearn import preprocessing
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor as KNN
from gplearn.genetic import SymbolicRegressor
from gplearn.functions import make_function
from gplearn.fitness import make_fitness
from sklearn.metrics import mean_squared_error, mean_absolute_error
from joblib import dump, load
from sklearn.model_selection import cross_val_score
import os

np.random.seed(2022)

#caculates the AARD
def mean_absolute_percentage_error(x_th,x_pr):
    return np.mean([100*abs(1-(x_pr[i]/x_th[i])) for i in range(len(x_th))])

#calculates D0 or eta0 if required
def calc_D0(T,d,state):
    if state=='e':
        D0=(5/16)*(T/math.pi)**(1/2)
    else:
        D0=(3/(8*d))*(T/math.pi)**(1/2)
    return D0

#prepares data for use in ML
def get_ML_data(state):
    data_s = pd.read_csv('data/Super_critical.csv')
    data_v = pd.read_csv('data/Vapour.csv')
    data_l = pd.read_csv('data/Liquid.csv')
    data_e = pd.read_csv('data/Viscosity.csv')
    new_state=""

    data=[]
    if "s" in state:
        data.append(data_s)
        new_state+="s"
    if "l" in state:
        data.append(data_l)
        new_state+="l"
    if "v" in state:
        data.append(data_v)
        new_state+="v"
    if "e" in state:
        data=[data_e]
        new_state="e"
    if "d" in state:
        data_l['Viscosity']=data_e['Viscosity'].values
        data=[data_l]
        new_state="d"
    if "c" in state:
        data_l['Viscosity']=data_e['Viscosity'].values
        data=[data_l]
        new_state="c"


    state=new_state

    if len(data)<1:
        print("No data found")
        exit()
    data = pd.concat(data,ignore_index=True)

    y_out=['Diffusion_Coefficient']

    if state=='e': y_out=['Viscosity']
    if state in'd': y_out.append('Viscosity')

    col_vals=y_out

    #removes unnecessary columns from input
    data = data.drop(columns=["Repulsive_Exponent","Attractive_Exponent","Temperature_Critical","Density_Critical"])

    #scales and prepares the data for using in ML
    X = data.drop(columns=col_vals)
    y = data[col_vals]
    y = np.log(y)
    if len(col_vals)==1:y=np.array(y).reshape(-1,1)
    X_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
    X = X_scaler.fit_transform(X)
    y_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
    y = y_scaler.fit_transform(y)

    if len(col_vals)==1:y=np.ravel(y)

    return X,y,X_scaler,y_scaler,state,y_out

#calculates the value of alpha
def get_alpha(n,m):
    d=n-m
    c=(n/d)*((n/m)**(m/d))
    return c*((1/(m-3))-(1/(n-3)))

#calculates r2
def get_r2(X_th,X_pr):
    res=sum((X_pr[j]-X_th[j])**2 for j in range(len(X_th)))
    tot=sum((y_val-np.mean(X_th))**2 for y_val in X_th)
    r2=1-res/tot
    return(float(r2))

#calculates SR values for X
def sub_eq(X,line):
    line=sympy.sympify(line.replace("$\eta$", "eta"))
    y=[]
    for j in range(len(X)):
        ans=line
        ans=ans.subs('D_0','(3/(8*rho))*((T/pi)**(1/2))')
        ans=ans.subs('eta_0','(5/16)*(T/pi)**(1/2)')
        ans=ans.subs('T',X[j][0])
        ans=ans.subs('rho',X[j][1])
        ans=ans.subs('alpha',X[j][2])
        ans=ans.subs('e',math.e)
        ans=ans.subs('pi',math.pi)
        try:
            ans=float(ans)
        except:
            print(f"Solution not found for {X[j]}. Value obtained={ans}")
        y.append(ans)
    return y

#writes the full output data to a file in the folder data_out
def write_output_data(X,y,y_pred,ML_type,Suffix):
    state=Suffix[-1]
    col_names=['Temperature','Density','alpha']
    OutFile=pd.DataFrame(X, columns=col_names)
    
    if state=='e':
        y_names=['Viscosity']
        y=[y]
        y_pred=[y_pred]
    elif state=='d':
        y_names=['Diffusion','Viscosity']
        y=np.transpose(y)
        y_pred=np.transpose(y_pred)
    else:
        y_names=['Diffusion']
        y=[y]
        y_pred=[y_pred]

    for id,name in enumerate(y_names):
        OutFile[f'Expermintal {name}']=y[id]
    for id,name in enumerate(y_names):
        OutFile[f'Predicted {name}']=y_pred[id]
   
    OutFile.to_csv(f'data_out/{ML_type}_{Suffix}.csv', index=False)
    return

#returns rescaled y output
def return_y(y,y_scaler):
    if len(np.shape(y))==1:
        y = np.exp(np.ravel(y_scaler.inverse_transform([y])))
    if len(np.shape(y))==2:
        y = np.exp(y_scaler.inverse_transform(y))
    return y

#performs the ANN algorithm
def do_ANN(X_train, X_test, y_train, y_test, X_scaler, y_scaler, Suffix):
    print("\nANN\n",flush=True)
    ML_model="ANN"
    for num_epochs in [3]: # log10 of the number of epochs to run
        for n_layers in [28]: #number of hidden layers
            any_good=[100,100,100] #value holder for ANN erformance metrics
            for iteration in range(1000):
                model = MLPRegressor(max_iter=10**num_epochs,
                                    activation="relu",
                                    hidden_layer_sizes=(n_layers),
                                    solver='lbfgs'
                                    )
                if iteration==0:
                    scores=cross_val_score(model,X_train,y_train, cv=10, scoring="neg_mean_squared_error")
                    print(f"CV10 performance: Mean MSE - {-1*np.mean(scores)}+{np.std(scores)}",flush=True) #Checks the cross validation of the first ANN algorithm
                model.fit(X_train,y_train)
                predictions = return_y(model.predict(X_test),y_scaler)

                score = [mean_absolute_percentage_error(y_test, predictions), mean_squared_error(y_test, predictions), mean_absolute_error(y_test, predictions)]
                if (score[0]< any_good[0]) or iteration==0:
                    best_model=model #records the best performing model score
                    any_good=score
            print(f"Model testing performance:  AARD - {score[0]}, MSE - {score[1]}, MAE - {score[2]}",flush=True) #prints testing performance
            dump(model, f"model_files/ANN_model_{n_layers}_{Suffix}.joblib") #dumps the best model
            write_output_data(X_scaler.inverse_transform(X_test),y_test,return_y(best_model.predict(X_test),y_scaler),ML_model,f"{Suffix}") #Makes output file
 
#performs the KNN algorithm
def do_KNN(X_train, X_test, y_train, y_test, X_scaler, y_scaler, Suffix):
    print("\nKNN\n",flush=True)
    ML_model="KNN"
    #KNN is purely deterministic, so additional repeats are not necessary
    model = KNN(n_neighbors=4,
                weights='distance',
                p=4)
    scores=cross_val_score(model,X_train,y_train, cv=10, scoring="neg_mean_squared_error") #Checks the cross validation of the KNN algorithm
    print(f"CV10 performance: Mean MSE - {-1*np.mean(scores)}+{np.std(scores)}",flush=True)
    model.fit(X_train,y_train)
    predictions = return_y(model.predict(X_test),y_scaler)

    dump(model, f"model_files/KNN_model_{Suffix}.joblib") #dumps the model
    score = [mean_absolute_percentage_error(y_test, predictions), mean_squared_error(y_test, predictions), mean_absolute_error(y_test, predictions)]
    print(f"Model testing performance:  AARD - {score[0]}, MSE - {score[1]}, MAE - {score[2]}",flush=True) #prints testing performance

    write_output_data(X_scaler.inverse_transform(X_test),y_test,return_y(model.predict(X_test),y_scaler),ML_model,f"{Suffix}") #Makes output file

#performs the SR algorithm
def do_SR(X_train, X_test, y_train, y_test, X_scaler, y_scaler, Suffix, input_args=[]):
    print("\nSR\n",flush=True)
    ML_model='SR'

    #defines the aard function for use inside the SR algorithm
    def _mape(y, y_pred, w):
        diffs = np.abs(np.divide((np.maximum(0.000001, y) - np.maximum(0.000001, y_pred)),
                                 np.maximum(0.000001, y)))
        return 100. * np.average(diffs, weights=w)
    mape = make_fitness(function=_mape,
                    greater_is_better=False)
    
    #defines the exp function for use inside the SR algorithm
    def _protected_exponent(x1):
        with np.errstate(over='ignore'):
            return np.where(np.abs(x1) < 100, np.exp(x1), 0.)
    exponent = make_function(function=_protected_exponent,name='exp',arity=1)

    
    #inputs passed on from Model_Machine_Learning.py
    if len(input_args)<1:n_repeats=input("How many tries? ")
    if len(input_args)>0:n_repeats=input_args[0]
    n_repeats=int(n_repeats)

    if len(input_args)<2:with_D0=input("With D0? ")
    if len(input_args)>1:with_D0=input_args[1]
    if with_D0 in ["1"]:with_D0=int(with_D0)

    state=Suffix[-1] #checks if D0 or eta0

    if with_D0:
        if state=='e':
            Suffix+="_eta0"
        else:
            Suffix+="_D0"

    print(Suffix,flush=True) #prints the Suffix, helps ensure that eta0 or D0 were passed/not passed.

    FileName=f"model_files/SR_values_{Suffix}.csv"

    # File=open(FileName,"a") #writes all the equations to SR_equations folder, appending to the previous ones
    try:
        col_vals=pd.read_csv(FileName)
    except:
        print("No old file found. A new one will be created",flush=True)
        col_vals=[[],[],[],[],[],[]]
        col_vals=pd.DataFrame(np.transpose(col_vals), columns=['AARD','MSE','MAE','seed','SR output','Equation'])
    
    X_train=X_scaler.inverse_transform(X_train)
    X_test=X_scaler.inverse_transform(X_test)
    y_train=np.exp(np.ravel(y_scaler.inverse_transform(y_train.reshape(1,-1))))

    if with_D0:
        for i in range(len(y_train)):
            D0=calc_D0(X_train[i][0],X_train[i][1],state)
            y_train[i]/=D0
        for i in range(len(y_test)):
            D0=calc_D0(X_test[i][0],X_test[i][1],state)
            y_test[i]/=D0


    feature_names=["T","rho","alpha"]

    set_of_functions=["add","sub","mul","div","sqrt","log",exponent]
    for i in range(n_repeats):
        print(f"Calculation number {i+1}/{n_repeats}",flush=True)
        random_number=randint(0,10**5)
        model = SymbolicRegressor(
                        population_size=5000,
                        # generations=50,
                        generations=5,
                        function_set=set_of_functions,
                        metric=mape,
                        parsimony_coefficient=0.3,
                        p_crossover=0.7,
                        p_subtree_mutation=0.1,
                        p_hoist_mutation=0.1,
                        p_point_mutation=0.1,
                        max_samples=0.9,
                        feature_names=feature_names,
                        verbose=1,
                        random_state=random_number
                        )
        model.fit(X_train,y_train)
        predictions = model.predict(X_test)

        # dump(model, f"model_files/SR_model_{Suffix}.joblib") #dumps the model. Commented out, as the full equation is saved
        score = [mean_absolute_percentage_error(y_test, predictions), mean_squared_error(y_test, predictions), mean_absolute_error(y_test, predictions)]
        print(score, random_number,'\n',flush=True) #prints the score
        print(model._program,'\n',flush=True) #prints final expression from SR

        #converts SR provided expression into a readable version
        converter = {
         'sub': lambda x, y : x - y,
         'div': lambda x, y : x/y,
         'mul': lambda x, y : x*y,
         'add': lambda x, y : x + y,
        'sqrt': lambda x    : (abs(x))**(1/2),
         'exp': lambda x    : math.e**x,
         'log': lambda x    : sympy.log(abs(x))
        }
        string=sympy.sympify(f"{model._program}", locals=converter)
        if with_D0:
            if state=='e':
                string=f"$\eta$_0*({string})"
            else:
                string=f"D_0*({string})"
        string=string.replace("2.71828182845905","e")
        
        print(string+'\n',flush=True) #print simplified final expression
        

        #Adds results to the result table
        new_vals=pd.DataFrame(score+[random_number]+[model._program]+[string]).transpose()
        new_vals.columns=col_vals.columns
        col_vals=pd.concat([col_vals,new_vals], ignore_index=True)
            
            
    col_vals=col_vals.sort_values(by=['AARD']) # sorts all equations by AARD
    col_vals.to_csv(FileName, index=False) # saves all equations
    if with_D0:
        for i in range(len(y_test)):
            D0=calc_D0(X_test[i][0],X_test[i][1],state)
            y_test[i]*=D0
    write_output_data(X_test,y_test,sub_eq(X_test,col_vals['Equation'][0]),ML_model,f"{Suffix}") #Makes output file for the best performing overall equation
 
def predict_new_file(File_to_predict,X_scaler,y_scaler,model,model_name):
    temp_lim=[0.45,1.5]
    dens_lim=[0.005,0.85]

    File_In=pd.read_csv(File_to_predict)
    File_bool = (File_In['Temperature'] > temp_lim[0]) & (File_In['Temperature'] < temp_lim[1]) & (File_In['Density'] > dens_lim[0]) & (File_In['Density'] < dens_lim[1])
    File_In=File_In[File_bool]
    
    X=File_In[['Temperature','Density','Alpha']]
    X=X_scaler.transform(X)
    if model_name!='SR':
        prediction=[model.predict(X)] #predict with ANN or KNN
        prediction=np.exp(np.ravel(y_scaler.inverse_transform(prediction)))
        X=X_scaler.inverse_transform(X)
    else:
        prediction=sub_eq(X_scaler.inverse_transform(X),model['Equation'][0]) #predict with SR
    X=np.concatenate((X.T,[File_In['Diffusion Coefficient'].to_numpy()],[prediction]))
    return pd.DataFrame(X.T,columns=['Temperature','Density','alpha','Expermintal Diffusion','Predicted Diffusion'])
    
    
    