# MLDiffusion
In the root directory the folders "data_out" and "plots" are required.
Libraries of _scikit_ 1.2.2 and _gplearn_ 0.4.2 are used.

To train a new model run **Model_Machine_Learning.py**, with the following user inputs:
- **ML type** - any ML types required can be named from ANN,KNN and SR in any order or comvination, e.g. ANN, ANNKNN, KNNSR etc.
- **state** and type of ML done - choose which properties to predict.
  - v,l,s in any order or combination, can predict the diffusion coefficient of vapour, liquid and super critical phases
  - e predicts liquid viscosity
- *SR Only* how many times to repeat the SR algorithm (if SR not used - use any number)
- **Regularize D** or $\eta$ with Chapman-Eskogg equations (0 - false/1 -true)
- example submit command **python Model_Machine_Learning.py ANNKNNSR slv 0 1**

To use a pretrained model - run **Predictions_Machine_Learning.py**, with the following user inputs:
- **state** and type of ML done - choose which properties to use.
  - v,l,s in any order or combination, can predict the diffusion coefficient of vapour, liquid and super critical phases
  - e predicts liquid viscosity
- **Regularize CE** for D or $\eta$ with Chapman-Eskogg equations (0 - false/1 -true)
- **Plot parity** plots (0 - false/1 -true). The parity plots are of the training files if no predict file is set.
- **Predict file** containing Mie data (0 - false/1 -true)
- **Predict input** from a user, with the user being able to input reduced temperature, reduced density and the Mie exponents in the terminal. (0 - false/1 -true)
- example submit command **python Predictions_Machine_Learning.py slv 1 1 0 1**

The file **input_Predictions.json** has 2 stored values.
- **Model_Folder** contains the name of a folder in model_files containing pretrained models
- **Additional_File** contains the name of the additional file to be openned.
