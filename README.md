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
* <"cross-validation"/"straightforward">- type of training
  * "cross-validation"- 10-fold cross validation.
  * "straightforward"
