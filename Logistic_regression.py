import random
import numpy as np

def rand_weights(size):
    dum= list()
    for i in range(size):
        dum.append(random.uniform(-1, 1))
    return np.array(dum)

def sigmoid(x):
    X=[(1 / (1 + np.exp(-z))) for z in x]
    return np.array(X)

def error(H, Y):
    er=0
    for i in range(len(H)):
        er = er + ((Y[i] * np.log(H[i])) + ((1 - Y[i]) * np.log(1 - H[i])))
    return round(er/len((H)),4)

def accuracy(class_f,Y_test):
    return (np.sum(class_f==Y_test)/len(Y_test))*100

class LogisticRegression():
    def __init__(self, learning_rate=0.05, maxIter=1000,error_ratio=0.01):
        self.__learning_rate = learning_rate
        self.__maxIter = maxIter
        self.__weigths = None
        self.__bias = 0
        self.__error_ratio=error_ratio

    def fit(self, X, Y):
        sample_size = np.array(X).shape[0]
        n_features = np.array(X).shape[1]
        self.__weigths = np.zeros(n_features)
        #self.__weigths = rand_weights(n_features)

        for i in range(self.__maxIter):
            linear = np.dot(X, self.__weigths) + self.__bias
            prediction = sigmoid(linear)

            if i % 100 == 0:
                print("Error =",error(prediction, Y),"    iteration :",i)

            dw = (1 / sample_size) * np.dot(X.T, (prediction - Y))
            db = (1 / sample_size) * np.sum(prediction - Y)

            self.__weigths = self.__weigths - self.__learning_rate * dw
            self.__bias = self.__bias - self.__learning_rate * db
            if self.__error_ratio > abs(error(prediction, Y)):
                break


    def predict(self, X_test):
        linear = np.dot(X_test, self.__weigths) + self.__bias
        Y_predicted = sigmoid(linear)
        class_f = [1 if y > 0.5 else 0 for y in Y_predicted]
        return class_f

    def get_weights(self):
        return self.__weigths