# Machine Learning Excercise (for the Data Mining graduate course poster competition).

Code related to implementing machine learning techniques to model the dark matter halo-galaxy connection.

CODE FILES
---------------------------
Artificial Neural Networks (ANN)
- ANN_DperKfold_class.py (Trains a deep neural network (artificial neural network with 2 hidden layers) to connect host halo properties with the galaxy luminosity of the occupying galaxy. Weights are found through gradient descent and K-fold validation is employed
- ANN_kfold.py (Similar to ANN_DperKfold, except now with the flexibility to change the number of neurons in a given layer, their activation function, employ batch training, alter step-size in gradient descent, and change loss function.)
- ILL_ANN_EVAL.py (Similar to above but now employing TensorFlow via Keras)

Gaussian Processes (GP)
- gp.py (Employ the GaPP gaussian processes program for Python to model fsig_8 vs z, delta matter vs z, delta_prime vs z, and f vs. z.)

Gaussian Mixtures Method (GMM)
- data_prep.py (Calculate the data covariance matrix)
- GM_means.py (Estimating the ideal # of means for GMM, via K_means)
- GM.py (Model data with GMM)


