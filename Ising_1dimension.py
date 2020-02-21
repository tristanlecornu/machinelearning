
import numpy as np
import scipy.sparse as sp
import warnings
import random
import pandas as pd
from math import *
from matplotlib import pyplot

from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline


############################ 1 Dimensions #################################################################

def ising_energies(states,L):
    #This function calculates the energies of the states in the nn Ising Hamiltonian
    J=np.zeros((L,L),)
    for i in range(L):
        J[i,(i+1)%L]-=1.0
        # compute energies
    E = np.einsum("...i,ij,...j->...",states,J,states)
    return E

############################# Build the design matrix ############################################
def design_matrix(states,N,L) :
    Xmatrix=np.zeros((N,1))
    for i in range(N):
        Q=0.0
        for j in range (L-1):
            Q += states[i][j]*states[i][j+1]
            #print(Q)
        Q +=  states[i][L-1]*states[i][0]
        #print("\n Q :\n {}\n".format(Q))
        Xmatrix[i] = Xmatrix[i] - Q
    #print(Xmatrix)
    return Xmatrix


# Neuronal Network 
def DNN():
    N_train = 10000
    L = 40
    maxdegree = 10
    states_train = np.random.choice([-1, 1], size=(N_train,L))
    from keras.utils import to_categorical
    tmp = ising_energies(states_train, L)[:,np.newaxis]+40

    Y_train = to_categorical(tmp)

    N_test = 5000
    states_test = np.random.choice([-1, 1], size=(N_test,L))
    Y_test = to_categorical(ising_energies(states_test, L)+40)
    from keras.models import Sequential
    from keras.layers import Dense, Activation
    from keras.regularizers import l2
    from keras.optimizers import Adam
    lmbd=1e-5
    eta =1e-5
    dnn = Sequential([
        Dense(1024, input_shape=(states_train.shape[1],), activation='relu', kernel_regularizer=l2(lmbd)),
        Dense(1024, activation='relu', kernel_regularizer=l2(lmbd)),
        Dense(1024, activation='relu', kernel_regularizer=l2(lmbd)),
        Dense(Y_train.shape[1], activation='softmax'),
    ])

    adam = Adam(learning_rate=eta)
    dnn.compile(loss='binary_crossentropy', optimizer=adam, metrics=['acc'])

    history = dnn.fit(states_train, Y_train, epochs=150, batch_size=32, validation_split=0.25)

    # Plot training & validation accuracy values
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    score = dnn.evaluate(states_test, Y_test, batch_size=32)
    print(score)
    return dnn, history, score


### define Ising model aprams
# system size
L=40
#Number of spins configuration
N=10000
# create 10000 random Ising states
np.random.seed(12)
states=np.random.choice([-1, 1], size=(N,L))
#print("states : \n {} \n".format(states))
# calculate Ising energies
energies=ising_energies(states,L)
#print("energies : \n {} \n".format(energies))
maxdegree = 10

DNN()

############################ Train Regression ##############################################

Xmatrix = design_matrix(states,N,L)

lamb = np.logspace(-7, 7, L)

mse_linear_train = np.zeros((L))
r2_linear_train = np.zeros((L))
mse_linear_test = np.zeros((L))
r2_linear_test = np.zeros((L))

mse_ridge_train = np.zeros((L))
r2_ridge_train = np.zeros((L))
mse_ridge_test = np.zeros((L))
r2_ridge_test = np.zeros((L))

mse_lasso_train = np.zeros((L))
r2_lasso_train = np.zeros((L))
mse_lasso_test = np.zeros((L))
r2_lasso_test = np.zeros((L))

for i,lmbd in enumerate(lamb):
    X_train, X_test, Y_train, Y_test = train_test_split(Xmatrix ,energies, test_size = 0.2, random_state=5)
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    model1 = Pipeline([("poly", PolynomialFeatures(degree=maxdegree)), 
                    ("linear", LinearRegression(fit_intercept=False))])

    model2 = Pipeline([("poly", PolynomialFeatures(degree=maxdegree)), 
                    ("ridge", Ridge(alpha=lmbd, fit_intercept=False))])

    model3 = Pipeline([("poly", PolynomialFeatures(degree=maxdegree)), 
                    ("lasso", Lasso(alpha=lmbd, fit_intercept=False))])

    # Fit the models
    fitted_linear_model= model1.fit(X_train, Y_train)
    fitted_ridge_model= model2.fit(X_train, Y_train)
    fitted_lasso_model= model3.fit(X_train, Y_train)

    # model evaluation for training set
    #print("coefficients linear model: {} \n".format(fitted_linear_model["linear"].coef_))
    #print("coefficients ridge model: {} \n".format(fitted_ridge_model["ridge"].coef_))
    #print("coefficients lasso model: {} \n".format(fitted_lasso_model["lasso"].coef_))

    # Linear regression 
    y_train_linear_predict = fitted_linear_model.predict(X_train)
    mse_linear_train[i] = mean_squared_error(Y_train, y_train_linear_predict)
    r2_linear_train[i] = r2_score(Y_train, y_train_linear_predict)
    #print("MSE linear train : {}\n".format(mse_linear_train[i]))
    #print("R2 linear train : {}\n".format(r2_linear_train[i]))

    # Ridge regression
    y_train_ridge_predict = fitted_ridge_model.predict(X_train)
    mse_ridge_train[i] = mean_squared_error(Y_train, y_train_ridge_predict)
    r2_ridge_train[i] = r2_score(Y_train, y_train_ridge_predict)
    #print("MSE ridge train : {}\n".format(mse_ridge_train[i]))
    #print("R2 ridge train : {}\n".format(r2_ridge_train[i]))

    # Lasso regression
    y_train_lasso_predict = fitted_lasso_model.predict(X_train)
    mse_lasso_train[i] = mean_squared_error(Y_train, y_train_lasso_predict)
    r2_lasso_train[i] = r2_score(Y_train, y_train_lasso_predict)
    #print("MSE lasso train : {}\n".format(mse_lasso_train[i]))
    #print("R2 lasso train : {}\n".format(r2_lasso_train[i]))

    ############################ Test Regression ###############################################
        
    # Linear regression
    y_test_linear_predict = fitted_linear_model.predict(X_test)
    # root mean square error of the model
    mse_linear_test[i] = mean_squared_error(Y_test, y_test_linear_predict)    
    # r-squared score of the model
    r2_linear_test[i] = r2_score(Y_test, y_test_linear_predict)
    #print ("MSE linear test : {}\n".format(mse_linear_test[i]))
    #print ("R2 linear test: {}\n".format(r2_linear_test[i]))

    # Ridge regression
    y_test_ridge_predict = fitted_ridge_model.predict(X_test)
    # root mean square error of the model
    mse_ridge_test[i] = mean_squared_error(Y_test, y_test_ridge_predict)    
    # r-squared score of the model
    r2_ridge_test[i] = r2_score(Y_test, y_test_ridge_predict)
    #print ("MSE ridge test : {}\n".format(mse_ridge_test[i]))
    #print ("R2 ridge test: {}\n".format(r2_ridge_test[i]))

    # Lasso regression
    y_test_lasso_predict = fitted_lasso_model.predict(X_test)
    # root mean square error of the model
    mse_lasso_test[i] = mean_squared_error(Y_test, y_test_lasso_predict)    
    # r-squared score of the model
    r2_lasso_test[i] = r2_score(Y_test, y_test_lasso_predict)
    #print ("MSE lasso test : {}\n".format(mse_lasso_test[i]))
    #print ("R2 lasso test: {}\n".format(r2_lasso_test[i]))


pyplot.plot(lamb, mse_linear_train, label="MSE linear_train", linewidth=3)
pyplot.plot(lamb, mse_linear_test, label="MSE linear_test")
pyplot.plot(lamb, mse_ridge_train, label="MSE ridge_train", linewidth=3)
pyplot.plot(lamb, mse_ridge_test, label="MSE ridge_test")
pyplot.plot(lamb, mse_lasso_train, label="MSE lasso_train", linewidth=3)
pyplot.plot(lamb, mse_lasso_test, label="MSE lasso_test")
pyplot.xscale('log')
pyplot.legend()
pyplot.show()


pyplot.plot(lamb, r2_linear_train, label="R2 linear_train", linewidth=3)
pyplot.plot(lamb, r2_linear_test, label="R2 linear_test")
pyplot.plot(lamb, r2_ridge_train, label="R2 ridge_train", linewidth=3)
pyplot.plot(lamb, r2_ridge_test, label="R2 ridge_test")
pyplot.plot(lamb, r2_lasso_train, label="R2 lasso_train", linewidth=3)
pyplot.plot(lamb, r2_lasso_test, label="R2 lasso_test")
pyplot.xscale('log')
pyplot.legend()
pyplot.show()