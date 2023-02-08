# polyhunt

My objective is to find the best fitting regularized weights for a linear model. My program is *vectorized* and reasonably efficient. I use the vector manipulation routines from numpy. These are implemented in compiled languages which are much more efficient than python I have created various synthetic datasets, each one being created by choosing a particular polynomial, sampling a subset of the x-axis, evaluating the polynomial at each point, and then adding a small amount of zero-mean Gaussian noise. My objective is to identify for each dataset the best fitting polynomial order that does not yield substantial
overfitting. 

I write an additional method that sweeps through each polynomial order (up to a given maximum). For each step, I then find the best weights and evaluate
accuracy (via rmse) and estimate the degree of overfitting. I then develop my own heuristics to identify the ”best” order as the one with the highest accuracy
achievable without significant overfitting. 

My program support the following commandline arguments:
- m - integer - polynomial order (or maximum in autofit mode)
- gamma - float - regularization constant (use a default of 0)
- trainPath - string - a filepath to the training data
- modelOutput - string - a filepath where the best fit parameters will be saved, if this is not supplied, then you do not have to output any model parameters
- autofit - boolean - a flag which when supplied engages the order sweeping loop, when this flag is false (or not supplied) I simply fit a polynomial of the given order and parameters. In either case, I save the best fit model to the file specified by the modelOutput path, and print the RMSE/order information to the screen.
