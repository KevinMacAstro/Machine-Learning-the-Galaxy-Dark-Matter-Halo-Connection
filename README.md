# MachineLearning

Code related to implementing machine learning techniques to model the dark-matter-halo--galaxy connection.

FILES
--------------------------
- ANN_GradTalk.key (Slides from talk given to other graduate students at the U regarding my exploration of the use of ANN to build a galaxy-halo occupation model)

CODE FILES
---------------------------
- ANN_DperKfold_class.py (Trains a deep neural network (artificial neural network with 2 hidden layers) to connect host halo properties with the galaxy luminosity of the occupying galaxy. Weights are fold through gradient descent and K-fold validation is employed. All coded from scratch)
- ANN_kfold.py (Similar to ANN_DperKfold, expect now with the flexibility to change the number of neurons in a given layer, their activation function, employ batch training, alter step-size in gradient descent, and change loss function. All coded from scratch.)
- gp.py (Employ the GaPP gaussian processes program for Python to model fsig_8 vs z, delta matter vs z, delta_prime vs z, and f vs. z.)
- 
