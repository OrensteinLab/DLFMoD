# DLFMoD
Deep Learning Framework for Motif Discovery

Requirements:
* Python version 3.6.8
* TensorFlow version 2.0.0
* keras version 2.1.6
* numpy version 1.17.4
* scikit-learn 0.22

Command line for training the model:

python main.py <"accuracy"/"motif"> <"cross-validation"/"straightforward"> <"simpleConvModel"> <"enhancer"> <32> <2048> <12> <1024> <"adam"> <20>

Inputs:
* <"accuracy"/"motif">
  * "accuracy"- The ROC curves of the best model and the average AUC score.
  * "motif"- Train and save the model for motif extraction step.
  
* <"cross-validation"/"straightforward"> - type of training
  * "cross-validation" - 10-fold cross validation.
  * "straightforward"
  
* <"simpleConvModel"> - Architecture name.

* <"enhancer"> - Data type.

* Model Hyperparameters:
  * <32> - Pooling size
  * <2048> - Number of Filters
  * <12> - Filter size
  * <1024> - Batch size
  * <"adam"> - Optimization algorithm
  * <20> - Number of epochs

Output:

* "accuracy" Outputs stored in images/
  * .png - ROC curve graph
 
* "motif" Outputs stored in images/ and models/
  * .png - ROC curve graph
  * .hdf5 - Model weights
  * .json - Trained model

Note- You need to add empty folders named "models" and "images" to the main folder.
