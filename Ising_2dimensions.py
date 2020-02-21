#%%
import os
import sys
import numpy as np
import pickle
from matplotlib import pyplot as plt
from math import *
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import random

from keras.utils import to_categorical

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.callbacks.callbacks import EarlyStopping

# Neural Network
def DNN(X_train, X_test, Y_train, Y_test, lmbd, eta, epochs=1000):
	dnn = Sequential([
		Dense(1024, input_shape=(X_train.shape[1],),
				activation='relu', kernel_regularizer=l2(lmbd)),
		Dense(Y_train.shape[1], activation='softmax'),
	])
	adam = Adam(learning_rate=eta)
	dnn.compile(loss='categorical_crossentropy',
				optimizer=adam, metrics=['acc'])

	history = dnn.fit(X_train, Y_train, epochs=epochs, batch_size=32,  validation_split=0.25, callbacks=[
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

def read_t (t=0.25,root="./"):
    if t > 0.:
        data = pickle.load(open(root+'Ising2DFM_reSample_L40_T=%.2f.pkl'%t,'rb'))
    else:
        data = pickle.load(open(root+'Ising2DFM_reSample_L40_T=0.25.pkl','rb'))
    return np.unpackbits(data).astype(int).reshape(-1,1600)
def read_X() :
    data = np.loadtxt("data.txt")
    data = np.vstack(data)
    return data
    

def drawX(x):
    # PREND UNE MATRICE DE X DE (40,40)
    plt.imshow(x)
    plt.show()

def meanConfig(x):
    bit_1 = 0
    bit_0 = 0
    #x = np.reshape(x,(1600,1))
    for j in range(x.shape[0]):
        #print("Taille de x : {}".format(x.shape))
        if x[j] == 1 :
            bit_1 += 1
        elif x[j] == 0 :
            bit_0 += 1
        else : 
            print("ERROR value of the data (bits [{},{}])".format(k,h))
        bit_total = x.shape[0]
        #print("Nombre de bits total : {} bits".format(bit_total))
        #print("Nombre de bits 1     : {} %".format(bit_1/bit_total*100))
        #print("Nombre de bits 0     : {} %".format(bit_0/bit_total*100))
    return bit_0, bit_1, bit_total

def classify_SDG(X,y):
    clf = linear_model.SGDClassifier(loss='log', penalty='l2', max_iter=100, shuffle=True, random_state=1, learning_rate='optimal').fit(X,y)
    print("SDG score : {}".format(clf.score(X, y)))
    return clf


def classify_liblinear(X, y):
    clf = linear_model.LogisticRegressionCV(solver='liblinear').fit(X,y)
    print("Logistic Regression score : {}".format(clf.score(X, y)))
    #print("parameters : {}".format(clf.get_params()))
    return clf

def main():
    #np.set_printoptions(threshold=sys.maxsize)

    stack = []
    for i,t in enumerate(np.arange(0.25,4.01,0.25)):
        y = np.ones(10000,dtype=int)
        if t > 2.25:
            y*=0
        stack.extend(list(y))

    y = np.array(stack)
    Y = np.vstack(y)

################################ Write X_matrix #####################################
    """stack = []
    
    for t in np.arange(0.25,4.01,0.25):
        stack.append(read_t(t,"./"))

    X = np.vstack(stack)
    N = 10000
    Xmatrix = np.zeros((160000,1))
    
    for i in range (16):
        print("Température : {} degres".format(np.arange(0.25,4.01,0.25)[i]))
        #print("Taille de x : {}".format(X.shape))
        bit0 = np.zeros((1,N))
        bit1 = np.zeros((1,N))

        for j in range(N):
            x = X[j+i*N]
       	    bit0[0][j],bit1[0][j],bitTot = meanConfig(x)
            #print(bit0[0][j],bit1[0][j],bitTot)
            Xmatrix[j+i*N][0]= np.float32(abs(bit0[0][j]-bit1[0][j])/bitTot)
    #np.savetxt("data.txt", Xmatrix)"""
########################################################################################
    Xmatrix = read_X()
    T1_X=Xmatrix[:10000, :]
    T1_Y=Y[:10000,:]
    T2_X=Xmatrix[10000:20000,:]
    T2_Y=Y[10000:20000,:]
    T3_X=Xmatrix[20000:30000,:]
    T3_Y=Y[20000:30000,:]
    T4_X=Xmatrix[30000:40000,:]
    T4_Y=Y[30000:40000,:]
    T5_X=Xmatrix[40000:50000,:]
    T5_Y=Y[40000:50000,:]
    T6_X=Xmatrix[50000:60000,:]
    T6_Y=Y[50000:60000,:]
    T7_X=Xmatrix[60000:70000,:]
    T7_Y=Y[60000:70000,:]
    T8_X=Xmatrix[70000:80000,:]
    T8_Y=Y[70000:80000,:]
    T9_X=Xmatrix[80000:90000,:]
    T9_Y=Y[80000:90000,:]
    T10_X=Xmatrix[90000:100000,:]
    T10_Y=Y[90000:100000,:]
    T11_X=Xmatrix[100000:110000,:]
    T11_Y=Y[100000:110000,:]
    T12_X=Xmatrix[110000:120000,:]
    T12_Y=Y[110000:120000,:]
    T13_X=Xmatrix[120000:130000,:]
    T13_Y=Y[120000:130000,:]
    T14_X=Xmatrix[130000:140000,:]
    T14_Y=Y[130000:140000,:]
    T15_X=Xmatrix[140000:150000,:]
    T15_Y=Y[140000:150000,:]
    T16_X=Xmatrix[150000:160000,:]
    T16_Y=Y[150000:160000,:]
    
    X_train, X_test, Y_train, Y_test = train_test_split(Xmatrix ,to_categorical(Y) , test_size = 0.2, random_state=5)
    DNN(X_train, X_test, Y_train, Y_test, 1e-5, 1e-5)
    

    #scaler = StandardScaler()
    #scaler.fit(X_train)
    #X_train_scaled = scaler.transform(X_train)
    #X_test_scaled = scaler.transform(X_test)
    #print(Xmatrix)


    ################################################### Accuracy #####################################################################

    """stack = []
    for t in np.arange(0.25,4.01,0.25):
        stack.append(read_t(t,"./"))

    X = np.vstack(stack)

    N = 10000
    threshold = 0.5
    acurateTab = np.zeros((16,N))
    
    meanBits = []
    for i in range (16):
        print("Température : {} degres".format(np.arange(0.25,4.01,0.25)[i]))
        print("Taille de x : {}".format(X.shape))
        bit0 = np.zeros((1,N))
        bit1 = np.zeros((1,N))

        for j in range(N):
            x = X[j+i*N]
            bit0[0][j],bit1[0][j],bitTot = meanConfig(x)
            if abs(bit0[0][j]-bit1[0][j])/100 >= threshold:
                acurateTab[i][j] = 1
        meanBits.append(np.mean(abs((bit0-bit1))))

    print("Les valeurs moyennes : \n {}".format(meanBits))
    print(acurateTab.shape)
    
    acuracy = []
    ACCURACY = 0.0

    totalCount = 0.0
    for k in range(16):
        count = 0.0
        for l in range(N):
            if acurateTab[k][l] == Y[k*N+l]:
                count += 1
        acuracy.append(count/N)
        totalCount += count
    ACCURACY = totalCount/(N*16)

    print("La précision par Température : \n {}".format(acuracy))
    print("La précision sur l'ensemble des données : {}".format(ACCURACY))"""
    ######################################################################################################################################

    X_train, X_test, Y_train, Y_test = train_test_split(Xmatrix ,Y , test_size = 0.2, random_state=5)
    clfc = classify_liblinear(X_train, Y_train)
    print("Test score : {}".format(clfc.score(X_test, Y_test)))
    print("T1 score : {}".format(clfc.score(T1_X, T1_Y)))
    print("T2 score : {}".format(clfc.score(T2_X, T2_Y)))
    print("T3 score : {}".format(clfc.score(T3_X, T3_Y)))
    print("T4 score : {}".format(clfc.score(T4_X, T4_Y)))
    print("T5 score : {}".format(clfc.score(T5_X, T5_Y)))
    print("T6 score : {}".format(clfc.score(T6_X, T6_Y)))
    print("T7 score : {}".format(clfc.score(T7_X, T7_Y)))
    print("T8 score : {}".format(clfc.score(T8_X, T8_Y)))
    print("T9 score : {}".format(clfc.score(T9_X, T9_Y)))
    print("T10 score : {}".format(clfc.score(T10_X, T10_Y)))
    print("T11 score : {}".format(clfc.score(T11_X, T11_Y)))
    print("T12 score : {}".format(clfc.score(T12_X, T12_Y)))
    print("T13 score : {}".format(clfc.score(T13_X, T13_Y)))
    print("T14 score : {}".format(clfc.score(T14_X, T14_Y)))
    print("T15 score : {}".format(clfc.score(T15_X, T15_Y)))
    print("T16 score : {}".format(clfc.score(T16_X, T16_Y)))
    
    clfc = classify_SDG(X_train, Y_train)
    print("Test score : {}".format(clfc.score(X_test, Y_test)))
    print("T1 score : {}".format(clfc.score(T1_X, T1_Y)))
    print("T2 score : {}".format(clfc.score(T2_X, T2_Y)))
    print("T3 score : {}".format(clfc.score(T3_X, T3_Y)))
    print("T4 score : {}".format(clfc.score(T4_X, T4_Y)))
    print("T5 score : {}".format(clfc.score(T5_X, T5_Y)))
    print("T6 score : {}".format(clfc.score(T6_X, T6_Y)))
    print("T7 score : {}".format(clfc.score(T7_X, T7_Y)))
    print("T8 score : {}".format(clfc.score(T8_X, T8_Y)))
    print("T9 score : {}".format(clfc.score(T9_X, T9_Y)))
    print("T10 score : {}".format(clfc.score(T10_X, T10_Y)))
    print("T11 score : {}".format(clfc.score(T11_X, T11_Y)))
    print("T12 score : {}".format(clfc.score(T12_X, T12_Y)))
    print("T13 score : {}".format(clfc.score(T13_X, T13_Y)))
    print("T14 score : {}".format(clfc.score(T14_X, T14_Y)))
    print("T15 score : {}".format(clfc.score(T15_X, T15_Y)))
    print("T16 score : {}".format(clfc.score(T16_X, T16_Y)))
    
    #print(Y_test)
    #print(X_test)

    
    plt.plot(Xmatrix, color="blue", label="experimental")
    plt.plot(Y, color="red", label="theory")
    plt.legend()
    plt.show()


    
if __name__ == "__main__":
    main()
