import sys
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.style.use('classic')
from glob import glob
from sklearn.model_selection import train_test_split
import json

from ML_functions import *

plt.rcParams['text.usetex'] = True

# python .py state D0 plot predict_file predict

# if plot is set to 1, a parity plot will be produced in plots/
# if predict_file is set to 1, user can input a file location from the ML_models folder to use
# if predict is set to 1, user can predict the required values by inputting them

state=""
if len(sys.argv)>1:state=sys.argv[1]
while len(state)<1: state=input("What state? ")

with_D0=""
if len(sys.argv)>2:with_D0=sys.argv[2]
while len(with_D0)<1:with_D0=input("With D0? ")
with_D0=int(with_D0)

plot=0
if len(sys.argv)>3:plot=int(sys.argv[3])

input_file=json.load(open('input_Predictions.json'))

predict_file=0
if len(sys.argv)>4:predict_file=int(sys.argv[4])
if predict_file: File_to_predict=glob(f"data/{input_file['Additional_File']}.csv")[0]

predict=0
if len(sys.argv)>5:predict=int(sys.argv[5])

additional_folder=input_file['Model_Folder']
if len(additional_folder)>0: additional_folder+='/'

X,y,X_scaler,y_scaler,state,y_out=get_ML_data(state,with_D0)

if len(y_out)>1:y_out=["Dual"]
Suffix=f"{y_out[0].split('_')[0]}_out_{state}"
if with_D0:
    if state=='e':
        Suffix+="_eta0"
    else:
        Suffix+="_D0"

all_models,all_model_names=get_ML_models(additional_folder,Suffix)

all_phase_files={'l':'Liquid','s':'Super_critical','v':'Vapour','e':'Viscosity'}
if plot or predict_file:
    for i,model in enumerate(all_model_names):   
        if not predict_file:
            Suffix_extra=""
            ML_vals=[]
            if model=='SR':
                for type in state:
                    ML_vals.append(predict_new_file(f'data/{all_phase_files[type]}.csv',X_scaler,y_scaler,all_models[i],model,state,with_D0))
                ML_vals_arr=pd.concat(ML_vals)
                train,test = train_test_split(ML_vals_arr, test_size=0.2, random_state=2020)
                train.to_csv(f'data_out/{model}_{Suffix}_train.csv',index=False)
                test.to_csv(f'data_out/{model}_{Suffix}_test.csv',index=False)
            else:
                for type in state:
                    ML_vals.append(predict_new_file(f'data/{all_phase_files[type]}.csv',X_scaler,y_scaler,all_models[i],model,state,with_D0))
                pd.concat(ML_vals).to_csv(f'data_out/{model}_{Suffix}.csv',index=False)
            if 's' in state:
                ML_vals.append(predict_new_file(f'data/Heat_Map_Data.csv',X_scaler,y_scaler,all_models[i],model,state,with_D0))
            col_names=ML_vals[-1].columns
            ML_vals=np.transpose(np.concatenate(ML_vals)).tolist()
        else:
            Suffix_extra=f"_{(File_to_predict.split('/')[-1]).split('.')[-2]}" #adds the name of the newly predicted file to the suffix
            ML_vals=predict_new_file(File_to_predict,X_scaler,y_scaler,all_models[i],model,state,with_D0)
            col_names=ML_vals.columns
            ML_vals=np.transpose(ML_vals.to_numpy()).tolist()

            
        ML_vals.append([])
        for i in range(len(ML_vals[-2])):
            ML_vals[-1].append(mean_absolute_percentage_error([ML_vals[-3][i]],[ML_vals[-2][i]]))
        ML_vals.append([])
        for i in range(len(ML_vals[-2])):
            ML_vals[-1].append(mean_absolute_error([ML_vals[-4][i]],[ML_vals[-3][i]]))
        errs=["MAE","MAPE"]
        for i in range(2):
            plot_Heatmap3D(ML_vals[1],ML_vals[0],ML_vals[2],ML_vals[-(i+1)],model,Suffix+Suffix_extra,f"_{errs[i]}")

        # ML_vals=np.transpose(ML_vals).tolist()
        num_cols=len(ML_vals)
        num_i=(num_cols-5)//2
        r2_val=r'$R^2$'
        et_al=r'$\it{et\ al.}$'
        train,test = train_test_split(np.transpose(ML_vals), test_size=0.2, random_state=2020)
        test=np.transpose(test)

    
        for i in range(3,3+num_i,2):
            fig = plt.figure(figsize=(6,6))
            ax = plt.gca()
            x_vals=ML_vals[i]
            y_vals=ML_vals[i+1]
            ax.plot(x_vals,y_vals,'ro',label=col_names[i].split()[1])
            ax.set_xscale("log")
            ax.set_yscale("log")
            x_min,x_max=ax.get_xlim()
            ax.plot([x_min,x_max],[x_min,x_max],'k-',label='parity')
            ax.set_xlim([x_min,x_max])
            ax.set_ylim([x_min,x_max])
            if predict_file:
                ax.set_xlabel(f'{col_names[i+1].split()[1]} from {Suffix_extra[1:]}')
                ax.set_ylabel(f'{col_names[i].split()[1]} from {model}')
            else:
                ax.set_xlabel(f'{col_names[i+1].split()[1]} from MD')
                ax.set_ylabel(f'{col_names[i].split()[1]} from {model}')
            ax.legend(loc='upper left')
            plt_ttl=f'{model} {state}_{i-3} AARD={mean_absolute_percentage_error(x_vals,y_vals):0.2f}\% R2={get_r2(x_vals,y_vals):0.4f}'
            plt_ttl=f'test {model} {state}_{i-3} AARD={mean_absolute_percentage_error(test[i],test[i+1]):0.2f}\% R2={get_r2(test[i],test[i+1]):0.4f}'
            print(plt_ttl)
            ax.set_title(plt_ttl)
            fig.savefig(f"plots/{model}_{Suffix}{Suffix_extra}_{i-3}.pdf", bbox_inches='tight')

    
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
    Suffix_extra=""
    ML_vals=[]
    for i,model in enumerate(all_model_names):  
        ML_out=predict_diff_values(X_input,[0],X_scaler,y_scaler,all_models[i],model,state,with_D0)
        print(f'{model} Prediciton: {ML_out.to_numpy()[-1][-1]:0.4f}')