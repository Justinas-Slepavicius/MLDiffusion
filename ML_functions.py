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
import sys
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.ndimage import gaussian_filter

np.random.seed(2022)
np.set_printoptions(threshold=sys.maxsize)

#caculates the AARD
def mean_absolute_percentage_error(x_th,x_pr):
    return np.mean([100*abs(1-(x_pr[i]/x_th[i])) for i in range(len(x_th))])

#calculates D0 or eta0 if required
def calc_CE(T,d,state):
    if state=='e':
        CE=(5/16)*(T/math.pi)**(1/2)
    else:
        CE=(3/(8*d))*(T/math.pi)**(1/2)
    return CE

#prepares data for use in ML
def get_ML_data(state,with_CE):
    data_s = pd.read_csv('data/Super_critical.csv')
    data_v = pd.read_csv('data/Vapour.csv')
    data_l = pd.read_csv('data/Liquid.csv')
    data_e = pd.read_csv('data/Viscosity.csv')
    new_state=""

    data=[]
    if "e" in state:
        data=[data_e]
        new_state="e"
    else:
        if "s" in state:
            data.append(data_s)
            new_state+="s"
        if "l" in state:
            data.append(data_l)
            new_state+="l"
        if "v" in state:
            data.append(data_v)
            new_state+="v"

    state=new_state

    if len(data)<1:
        print("No data found")
        exit()
    data = pd.concat(data,ignore_index=True)

    y_out=['Diffusion_Coefficient']

    if state=='e': y_out=['Viscosity']

    col_vals=y_out
    if 'Diffusion_Coefficient_Error' in data.columns: data=data.drop(columns=['Diffusion_Coefficient_Error'])
    
    #removes unnecessary columns from input
    data = data.drop(columns=["Repulsive_Exponent","Attractive_Exponent","Temperature_Critical","Density_Critical"])

    #scales and prepares the data for using in ML
    X = data.drop(columns=col_vals)
    y = data[col_vals].values.tolist()
    if with_CE:
        for i in range(len(y)):
            CE=calc_CE(X.values.tolist()[i][0],X.values.tolist()[i][1],state)
            y[i][0]/=CE
    else:
        y = np.log(y)
    if len(col_vals)==1:y=np.array(y).reshape(-1,1)
        
    X_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
    X = X_scaler.fit_transform(X)
    y_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
    y = y_scaler.fit_transform(y)

    if len(col_vals)==1:y=np.ravel(y)

    return X,y,X_scaler,y_scaler,state,y_out

#prepares the models to use in predictions
def get_ML_models(additional_folder,Suffix):
    all_models=[]
    all_model_names=[]

    try:
        model_ANN = load(f"model_files/{additional_folder}ANN_model_28_{Suffix}.joblib")
        all_models.append(model_ANN)
        all_model_names.append("ANN")
    except:
        pass
    try:
        model_KNN = load(f"model_files/{additional_folder}KNN_model_{Suffix}.joblib")
        all_models.append(model_KNN)
        all_model_names.append("KNN")
    except:
        pass
    try:
        model_SR  = pd.read_csv(f"model_files/{additional_folder}SR_values_{Suffix}.csv")
        all_models.append(model_SR)
        all_model_names.append("SR")
    except:
        pass
    
    if len(all_model_names)<1:
        print("No models found")
        exit()
    
    return all_models,all_model_names

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
def return_y(y,y_scaler,with_CE):
    y = np.ravel(y_scaler.inverse_transform(y))
    if not with_CE:
        y = np.exp(y)
    return y

#performs the ANN algorithm
def do_ANN(X_train, X_test, y_train, y_test, X_scaler, y_scaler, with_CE, Suffix):
    print("\nANN\n",flush=True)
    X_test_scaled=X_scaler.inverse_transform(X_test)
    ML_model="ANN"
    for num_epochs in [4]: # log10 of the number of epochs to run
        for n_layers in [28]: #number of hidden layers
            any_good=[100,100,100] #value holder for ANN erformance metrics
            for iteration in range(100):
                model = MLPRegressor(max_iter=10**num_epochs,
                                    activation="relu",
                                    hidden_layer_sizes=(n_layers),
                                    solver='lbfgs'
                                    )
                if iteration==0:
                    scores=cross_val_score(model,X_train,y_train, cv=10, scoring="neg_mean_squared_error")
                    print(f"CV10 performance: Mean MSE - {-1*np.mean(scores)}+{np.std(scores)}",flush=True) #Checks the cross validation of the first ANN algorithm
                model.fit(X_train,y_train)
                predictions = return_y([model.predict(X_test)],y_scaler,with_CE)

                if with_CE:
                    state=Suffix.split('_')[-2]
                else:
                    state=Suffix.split('_')[-1]
                if with_CE:
                    for i in range(len(predictions)):
                        CE=calc_CE(X_test_scaled[i][0],X_test_scaled[i][1],state)
                        predictions[i]*=CE
                
                score = [mean_absolute_percentage_error(y_test, predictions), mean_squared_error(y_test, predictions), mean_absolute_error(y_test, predictions)]
                if (score[0]< any_good[0]) or iteration==0:
                    best_model=model #records the best performing model score
                    any_good=score
            print(f"Model testing performance:  AARD - {score[0]}, MSE - {score[1]}, MAE - {score[2]}",flush=True) #prints testing performance
            dump(model, f"model_files/ANN_model_{n_layers}_{Suffix}.joblib") #dumps the best model
            write_output_data(X_scaler.inverse_transform(X_test),y_test,return_y([best_model.predict(X_test)],y_scaler,with_CE),ML_model,f"{Suffix}") #Makes output file
 
#performs the KNN algorithm
def do_KNN(X_train, X_test, y_train, y_test, X_scaler, y_scaler, with_CE, Suffix):
    print("\nKNN\n",flush=True)
    ML_model="KNN"
    X_test_scaled=X_scaler.inverse_transform(X_test)
    #KNN is purely deterministic, so additional repeats are not necessary
    model = KNN(n_neighbors=4,
                weights='distance',
                p=4)
    scores=cross_val_score(model,X_train,y_train, cv=10, scoring="neg_mean_squared_error") #Checks the cross validation of the KNN algorithm
    print(f"CV10 performance: Mean MSE - {-1*np.mean(scores)}+{np.std(scores)}",flush=True)
    model.fit(X_train,y_train)
    predictions = return_y([model.predict(X_test)],y_scaler, with_CE)

    if with_CE:
        state=Suffix.split('_')[-2]
    else:
        state=Suffix.split('_')[-1]
    if with_CE:
        for i in range(len(predictions)):
            CE=calc_CE(X_test_scaled[i][0],X_test_scaled[i][1],state)
            predictions[i]*=CE

    dump(model, f"model_files/KNN_model_{Suffix}.joblib") #dumps the model
    score = [mean_absolute_percentage_error(y_test, predictions), mean_squared_error(y_test, predictions), mean_absolute_error(y_test, predictions)]
    print(f"Model testing performance:  AARD - {score[0]}, MSE - {score[1]}, MAE - {score[2]}",flush=True) #prints testing performance

    write_output_data(X_scaler.inverse_transform(X_test),y_test,return_y([model.predict(X_test)],y_scaler,with_CE),ML_model,f"{Suffix}") #Makes output file

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

    if len(input_args)<2:with_CE=input("With CE regularisation? ")
    if len(input_args)>1:with_CE=input_args[1]
    if with_CE in ["1"]:with_CE=int(with_CE)

    if with_CE:
        state=Suffix.split('_')[-2]
    else:
        state=Suffix.split('_')[-1]

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
    y_train=np.ravel(y_scaler.inverse_transform(y_train.reshape(1,-1)))
    if not with_CE:
        y_train=np.exp(y_train)

    feature_names=["T","rho","alpha"]

    set_of_functions=["add","sub","mul","div","sqrt","log",exponent]
    for i in range(n_repeats):
        print(f"Calculation number {i+1}/{n_repeats}",flush=True)
        random_number=randint(0,10**5)
        model = SymbolicRegressor(
                        population_size=5000,
                        generations=50,
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
        if with_CE:
            for i in range(len(predictions)):
                CE=calc_CE(X_test[i][0],X_test[i][1],state)
                predictions[i]*=CE

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
        if with_CE:
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
    if with_CE:
        for i in range(len(y_test)):
            CE=calc_CE(X_test[i][0],X_test[i][1],state)
            y_test[i]*=CE
    write_output_data(X_test,y_test,sub_eq(X_test,col_vals['Equation'][0]),ML_model,f"{Suffix}") #Makes output file for the best performing overall equation
 
def predict_new_file(File_to_predict,X_scaler,y_scaler,model,model_name,state,with_CE):
    temp_lim=[0.45,1.5]
    dens_lim=[0.005,0.85]


    File_In=pd.read_csv(File_to_predict)
    if 'Diffusion_Coefficient_Errors' in File_In.columns: File_In.drop(columns=['Diffusion_Coefficient_Errors'])
    File_bool = (File_In['Temperature'] > temp_lim[0]) & (File_In['Temperature'] < temp_lim[1]) & (File_In['Density'] > dens_lim[0]) & (File_In['Density'] < dens_lim[1])
    File_In=File_In[File_bool]

    X=File_In[['Temperature','Density','Alpha']]
    if 'Diffusion_Coefficient' in File_In.columns:File_In.rename(columns={'Diffusion_Coefficient': 'Diffusion Coefficient'}, inplace=True)
    if state=='e':
        test_data=File_In['Viscosity'].to_numpy().copy()
    else:
        test_data=File_In['Diffusion Coefficient'].to_numpy().copy()
    test_og=test_data.copy()
    for i in range(len(test_data)):
        CE=calc_CE(X.values.tolist()[i][0],X.values.tolist()[i][1],state)
        test_data[i]/=CE
    
    X=X_scaler.transform(X)
    if state=='e':
        y=File_In['Viscosity'].to_numpy()
    else:
        y=File_In['Diffusion Coefficient'].to_numpy()
      
    return predict_diff_values(X,y,X_scaler,y_scaler,model,model_name,state,with_CE)

def predict_diff_values(X,y,X_scaler,y_scaler,model,model_name,state,with_CE):    
    if model_name!='SR':
        prediction=[model.predict(X)] #predict with ANN or KNN
        prediction=return_y(prediction,y_scaler,with_CE)
        X=X_scaler.inverse_transform(X)
        if with_CE:
            for i in range(len(prediction)):
                CE=calc_CE(X[i][0],X[i][1],state)
                prediction[i]*=CE
    else:
        X=X_scaler.inverse_transform(X)
        prediction=sub_eq(X,model['Equation'][0]) #predict with SR
    

    X=np.concatenate((X.T,[y],[prediction]))


    transport=['Diffusion','Viscosity']
    tr=0
    if state=='e':tr=1
    
    return pd.DataFrame(X.T,columns=['Temperature','Density','alpha',f'Expermintal {transport[tr]}',f'Predicted {transport[tr]}'])

def plot_Heatmap(x,y,errors,model,Suffix,Suffix_extra):
    Suffix_extra_og=Suffix_extra
    for con in [9,5,1]:
        Suffix_extra=f'{Suffix_extra_og}_{con}'
        num_points=10*con
        y_mesh, x_mesh = np.meshgrid(np.linspace(min(y), max(y), num_points), np.linspace(min(x), max(x), num_points))
        z_mesh = 0*x_mesh*y_mesh
        z_count= 0*x_mesh*y_mesh
        for x_val,y_val,z_val in zip(x,y,errors):
            for j in range(0,len(x_mesh)):
                if x_mesh[j][0]>x_val:
                    for k in range(0,len(y_mesh[0])):
                        if y_mesh[0][k]>y_val:
                            for l in range(-(con//2),(con//2)+1):
                                for m in range(-(con//2),(con//2)+1):
                                    if l+j-1>=0 and m+k-1>=0:
                                        if l+j-1<len(z_mesh) and m+k-1<len(z_mesh[0]):
                                            z_mesh[l+j-1][m+k-1]+=z_val
                                            z_count[l+j-1][m+k-1]+=1
                            break
                    break
        for i in range(len(z_mesh)):
            for j in range(len(z_mesh[i])):
                if z_count[i][j]!=0:
                    z_mesh[i][j]/=z_count[i][j]
                
        z_mesh=gaussian_filter(z_mesh, sigma=con**3-1,radius=con)
        # np.savetxt(f'z_mesh_{Suffix_extra}.csv',z_mesh,delimiter=",")
        # np.savetxt(f'z_count_{Suffix_extra}.csv',z_count,delimiter=",")

        z_min, z_max = 0, np.abs(z_mesh).max()

        fig, ax = plt.subplots()

        c = ax.pcolormesh(x_mesh, y_mesh, z_mesh, cmap='Reds', vmin=z_min, vmax=z_max)
        text=Suffix_extra.split('_')[1:]
        ax.set_title(f'Heatmap of {text[0]} for {model} with convolution={text[-1]}')
        if 'T' in text:
            ax.set_ylabel(r'$T^*$',size=20)
            if 'rho' in text:
                ax.set_xlabel(fr'$\rho^*$',size=20)
            else:
                ax.set_xlabel(fr'$\alpha$',size=20)
        else:
            ax.set_ylabel(r'$\alpha$',size=20)
            ax.set_xlabel(r'$\rho}^*$',size=20)
            
        ax.axis([x_mesh.min(), x_mesh.max(), y_mesh.min(), y_mesh.max()])
        cbar=fig.colorbar(c, ax=ax)
        
        cbar.ax.get_yaxis().labelpad = 15
        cbar.ax.set_ylabel(f'{text[0]}', rotation=270)

        fig.savefig(f"plots/{model}_{Suffix}{Suffix_extra}.pdf", bbox_inches='tight')
        plt.close(fig)

def plot_Heatmap3D(x,y,z,errors,model,Suffix,Suffix_extra):
    text=Suffix_extra.split('_')[1:]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plt.scatter(x,y,zs=z,s=10,c=errors,cmap=mpl.colormaps['hot_r'],linewidths=0.5)
    color_map = mpl.cm.ScalarMappable(cmap=mpl.cm.hot_r)
    color_map.set_array(errors)
    cax = plt.axes([0.9, 0.1, 0.02, 0.8])
    cbar=plt.colorbar(color_map,cax=cax)
    cbar.ax.get_yaxis().labelpad = 15
    cbar.ax.set_ylabel(f'{text[0]}', rotation=270)
    ax.set_title(f'3D Heatmap of {text[0]} for {model} ')
    ax.set_xlabel(r'$\rho^*$',size=20)
    ax.set_ylabel(r'$T^*$',size=20)
    ax.set_zlabel(r'$\alpha$',size=20)
    ax.set_xlim(min(x),max(x))
    ax.set_ylim(min(y),max(y))
    ax.set_zlim(min(z),max(z))

    fig.savefig(f"plots/{model}_{Suffix}{Suffix_extra}_3D.pdf", bbox_inches='tight')
    plt.close(fig)