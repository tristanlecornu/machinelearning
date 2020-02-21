#%%
import os
import sys
import random
import numpy as np
import csv
import pickle
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

#%%
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.callbacks.callbacks import EarlyStopping
def DNN(lmbd, eta):
	dnn = Sequential([
		Dense(n_hidden_neurons, input_shape=(n_features,),
				activation='sigmoid', kernel_regularizer=l2(lmbd)),
		Dense(n_categories, activation='sigmoid'),
	])
	adam = Adam(learning_rate=eta)
	dnn.compile(loss='mean_squared_error',
				optimizer=adam, metrics=['acc'])

	history = dnn.fit(X_train_scaled, Y_train, epochs=epochs, batch_size=32,  validation_split=0.25, callbacks=[
		EarlyStopping(monitor='val_loss', min_delta=1e-5, patience=10,
						verbose=0, mode='auto', baseline=None, restore_best_weights=True)
	])

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

	score = dnn.evaluate(X_test, Y_test, batch_size=32)
	print(score)
	return dnn

def classify(X_, Y_):
	clf = linear_model.LogisticRegression()
	clf.fit(X_, Y_)
	return clf


if __name__ == "__main__":

	np.set_printoptions(threshold=sys.maxsize)

	FILE_NAME = 'save_data_pulsar.p'

	if os.path.exists(FILE_NAME):
		data = pickle.load(open(FILE_NAME, "rb"))
	else:
		print("DATA file {} doesnt exists any more. Sad ..... Lunch it and back later.".format(
			FILE_NAME))
		sys.exit()

	# COLLECT DATA
	DATA = pd.DataFrame(data)
	print(" -> DATA BASE (debug) :: \n{}\n".format(DATA))

	# DATA IN 'DataFrame'
	X = DATA.iloc[1:, 0:7]
	Y = DATA.iloc[1:, 8]

	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.5)

	# SCALLING DATA
	scaler = StandardScaler()
	scaler.fit(X_train)
	X_train_scaled = scaler.transform(X_train)
	X_test_scaled = scaler.transform(X_test)

	# CLASSIFICATION
	clf = classify(X_train_scaled,Y_train)
	Y_pred = clf.predict(X_test_scaled)#.ravel()

	print(" (debug) :: Taille de la matrice Ytest  : {}".format(Y_test.shape))
	print(" (debug) :: Taille de la matrice Y_pred : {}".format(Y_pred.shape))

	#print("PROBA        : \n {}".format(clf.predict_proba(Xtest)))
	print("SCORE (logistic) : {}".format(clf.score(X_test,Y_pred)))


	# NEURAL NETWORKS

	epochs = 1000
	batch_size = 100
	n_inputs, n_features = X_train_scaled.shape
	n_hidden_neurons = 50
	n_categories = 1

	eta = 0.00001
	lmbd = 0.00001
	dnn = DNN(lmbd, eta)