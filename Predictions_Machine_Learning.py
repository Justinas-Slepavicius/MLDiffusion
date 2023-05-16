import sys
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.style.use('classic')
from glob import glob

from ML_functions import *

# python .py state D0 plot predict_file predict
# if plot is set to 1, a parity plot will be produced in plots/
# if predict_file is set to 1, user can input a file location from the ML_models folder to use
# if predict is set to 1, user can predict the required values by inputting them

state=""
if len(sys.argv)>1:state=sys.argv[1]
while len(state)<1: state=input("What state? ")

with_D0=""
if len(sys.argv)>2:with_D0=sys.argv[2]
while len(state)<1:with_D0=input("With D0? ")
with_D0=int(with_D0)

plot=0
if len(sys.argv)>3:plot=int(sys.argv[3])

predict_file=0
if len(sys.argv)>4:predict_file=int(sys.argv[4])
if predict_file: correct_file=0
while not correct_file:
    File_to_predict=input("Choose file to use ") #allows the user to choose a file for which to predict values
    File_to_predict=glob(f"data/{File_to_predict}*")
    correct_file=input(f"Is the file {File_to_predict[0]}? ")
    if correct_file in '1yes':
        correct_file=1
        File_to_predict=File_to_predict[0]
    else:
        correct_file=0

predict=0
if len(sys.argv)>5:predict=int(sys.argv[5])

additional_folder=input("Additional folder? ") #Allows for entry of a specific folder in model_files/ that contains models required
if len(additional_folder)>0: additional_folder+='/'

X,y,X_scaler,y_scaler,state,y_out=get_ML_data(state)

if len(y_out)>1:y_out=["Dual"]
Suffix=f"{y_out[0].split('_')[0]}_out_{state}"
if with_D0:
    if state=='e':
        Suffix_SR="_eta0"
    else:
        Suffix_SR="_D0"
else:
    Suffix_SR=""


model_ANN = load(f"model_files/{additional_folder}ANN_model_28_{Suffix}.joblib")
model_KNN = load(f"model_files/{additional_folder}KNN_model_{Suffix}.joblib")
model_SR  = pd.read_csv(f"model_files/{additional_folder}SR_values_{Suffix}{Suffix_SR}.csv")

all_models=[model_ANN,model_KNN,model_SR]
all_model_names=["ANN","KNN","SR"]

if plot or predict_file:
    for i,model in enumerate(all_model_names):
        if model=="SR":            
            Suffix_extra=Suffix_SR
        else:
            Suffix_extra=""
        
        try:
            ML_vals=pd.read_csv(f"data_out/{model}_{Suffix}{Suffix_extra}.csv")
        except:
            print(f"No data_out/{model}_{Suffix}.csv file found")
            continue
    
        if predict_file:
            ML_vals=predict_new_file(File_to_predict,X_scaler,y_scaler,all_models[i],model)
        
        col_names=ML_vals.columns
        
        if predict_file:
            Suffix_extra+=f"_{(File_to_predict.split('/')[-1]).split('.')[-2]}" #adds the name of the newly predicted file to the suffix
            
    
        for i in range((len(col_names)-3)//2):
            fig = plt.figure()
            ax = plt.gca()
            x_vals=ML_vals[col_names[3+i]]
            y_vals=ML_vals[col_names[3+((len(col_names)-3)//2)+i]]
            ax.plot(x_vals,y_vals,'ro',label=col_names[3+i].split()[1])
            ax.set_xscale("log")
            ax.set_yscale("log")
            x_min,x_max=ax.get_xlim()
            ax.plot([x_min,x_max],[x_min,x_max],'k-',label='parity')
            ax.set_xlim([x_min,x_max])
            ax.set_ylim([x_min,x_max])
            ax.set_xlabel(col_names[3+i])
            ax.set_ylabel(col_names[3+((len(col_names)-3)//2)+i])
            ax.legend(loc='upper left')
            ax.set_title(f'{model} {state}_{i} AARD={mean_absolute_percentage_error(x_vals,y_vals):0.2f}% R2={get_r2(x_vals,y_vals):0.4f}')
            fig.savefig(f"plots/{model}_{Suffix}{Suffix_extra}_{i}.pdf", bbox_inches='tight')

    
while predict:
    temp = input("Temperature:") #Input temperature. if a non-numerical value is put in - loop stops
    try:
        float(temp)
    except ValueError:
        break
    dens = float(input("Density:")) #Input other values
    l_rep = float(input("Repulsive Exponent:"))
    l_att = float(input("Attractive Exponent:"))
    X_input= np.array([temp,dens,get_alpha(l_rep,l_att)]).reshape(1,-1)
    X_input=X_scaler.transform(X_input)
    prediction=[model_ANN.predict(X_input)] #predict with ANN
    print(f"ANN predicted value: {np.exp(y_scaler.inverse_transform(prediction))[0][0]}")
    prediction=[model_KNN.predict(X_input)] #predict with KNN
    print(f"KNN predicted value: {np.exp(y_scaler.inverse_transform(prediction))[0][0]}")
    if with_D0:
        if state=='e':
            Suffix+="_eta0"
        else:
            Suffix+="_D0"
    Best_SR=pd.read_csv(f"model_files/{additional_folder}SR_values_{Suffix}{Suffix_SR}.csv")
    prediction=sub_eq(X_scaler.inverse_transform(X_input),Best_SR['Equation'][0]) #predict with SR
    print(f"SR predicted value: {prediction[0]}")