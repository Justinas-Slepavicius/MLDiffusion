# MLDiffusion
In the root directory the folders "data_out" and "plots" are required.
Libraries of scikit 1.2.2 and gplearn 0.4.2

To train a new model run Model_Machine_Learning.py, with the following user inputs:
- ML type - any ML types required can be named from ANN,KNN and SR in any order or comvination, e.g. ANN, ANNKNN, KNNSR etc.
- state and type of ML done - choose which properties to predict.
  - v,l,s in any order or combination, can predict the diffusion coefficient of vapour, liquid and super critical phases
  - e predicts liquid viscosity
- *SR Only* how many times to repeat the SR algorithm (if SR not used - use any number)
- Regularize D or $\eta$ with Chapman-Eskogg equations (0 - false/1 -true)
example submit command \textbf{python Model_Machine_Learning ANNKNNSR slv 0 1}

To use a pretrained model - run Predictions_Machine_Learning.py, with the following user inputs: state D0 plot predict_file predict
- state and type of ML done - choose which properties to use.
  - v,l,s in any order or combination, can predict the diffusion coefficient of vapour, liquid and super critical phases
  - e predicts liquid viscosity
  - d predicts viscosity and diffusion coefficient simultaneously (does not work with SR)
- *SR Only* Regularize D or $\eta$ with Chapman-Eskogg equations (0 - false/1 -true)
- Plot parity plots (0 - false/1 -true)
- Predict data from a file (0 - false/1 -true)
- predict singular user inputs (0 - false/1 -true)

The file will also ask for an addition folder where ML models are stored:
- Leave empty if the ML model you want to use is recently trained.
- Use "Slepavicius" for the pretrained models found in the manuscript

Can make your own folder in the data directory to store pretrained models
