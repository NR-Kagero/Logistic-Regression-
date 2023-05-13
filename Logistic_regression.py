import random
import numpy as np

class LogisticRegression():
    def __init__(self, learning_rate=0.05, maxIter=1000, error_ratio=0.0001, L1=0, batch_size=1, beta_1=0.9, beta_2=0.9,
                 epsilon=0.5):
        self.__epsilon = epsilon
        self.__beta_2 = beta_2
        self.__beta_1 = beta_1
        self.__learning_rate = learning_rate
        self.__maxIter = maxIter
        self.__weigths = None
        self.__bias = 0
        self.__error_ratio = error_ratio
        self.__L1 = L1
        self.__batch_size = batch_size

    def fit(self, X, Y, optimizer="None"):
        sample_size = np.array(X).shape[0]
        n_features = np.array(X).shape[1]
        # self.__weigths = np.zeros(n_features)
        self.__weigths = self._rand_weights(n_features, "rand")

        Error = 1
        if self.__batch_size > 1:
            total = sample_size // self.__batch_size
            batches = np.random.choice(len(X), size=[sample_size // self.__batch_size, self.__batch_size],
                                       replace=False)
            batched_data = np.array(X[batches])
            batched_labels = np.array(Y[batches])
        else:
            total = 1
            batched_data = np.array(X).reshape(1, sample_size, n_features)
            batched_labels = np.array(Y).reshape(1, -1)

        m_dw = np.zeros_like(self.__weigths)
        v_dw = np.zeros_like(self.__weigths)
        m_db = 0
        v_db = 0
        t = 0

        for i in range(self.__maxIter):
            epoch_bar = tqdm(total=total, desc=f"Epochs {i + 1}/{self.__maxIter}")
            for I in range(len(batched_data)):
                linear = np.dot(batched_data[I], self.__weigths) + self.__bias
                prediction = self._sigmoid(linear)
                x = np.array(batched_data[I])
                y = np.array(batched_labels[I]).reshape(-1)
                dw = (1 / self.__batch_size) * np.dot(x.T, (prediction - y)) + (self.__L1 / (2 * sample_size)) * np.sum(np.abs(self.__weigths))
                db = (1 / self.__batch_size) * np.sum(prediction - y)
                t += 1
                if optimizer == "adam":
                    m_dw = self.__beta_1 * m_dw + (1 - self.__beta_1) * dw
                    m_db = self.__beta_1 * m_db + (1 - self.__beta_1) * db
                    v_dw = self.__beta_2 * v_dw + (1 - self.__beta_2) * (dw ** 2)
                    v_db = self.__beta_2 * v_db + (1 - self.__beta_2) * (db ** 2)
                    v_dw_hat = v_dw / (1 - self.__beta_2 ** t)
                    v_db_hat = v_db / (1 - self.__beta_2 ** t)
                    m_dw_hat = m_dw / (1 - self.__beta_1 ** t)
                    m_db_hat = m_db / (1 - self.__beta_1 ** t)
                    self.__weigths = self.__weigths - self.__learning_rate * m_dw_hat / (
                            np.sqrt(v_dw_hat) + self.__epsilon)
                    self.__bias = self.__bias - self.__learning_rate * m_db_hat / (np.sqrt(v_db_hat) + self.__epsilon)
                elif optimizer == "rms":
                    v_dw = self.__beta_2 * v_dw + (1 - self.__beta_2) * (dw ** 2)
                    v_db = self.__beta_2 * v_db + (1 - self.__beta_2) * (db ** 2)
                    v_dw_hat = v_dw / (1 - self.__beta_2 ** t)
                    v_db_hat = v_db / (1 - self.__beta_2 ** t)
                    self.__weigths = self.__weigths - self.__learning_rate * dw / (np.sqrt(v_dw_hat) + self.__epsilon)
                    self.__bias = self.__bias - self.__learning_rate * db / (np.sqrt(v_db_hat) + self.__epsilon)
                elif optimizer == "None":
                    self.__weigths = self.__weigths - self.__learning_rate * dw
                    self.__bias = self.__bias - self.__learning_rate * db
                Error = abs(self._error(prediction, y, self.__L1))
                epoch_bar.update(1)
                epoch_bar.set_postfix({'Accuracy ': f'{1 - Error:.3f}'})
                if self.__error_ratio > Error:
                    print(Error)
                    return
            del epoch_bar

    def predict(self, X_test):
        linear = np.dot(X_test, self.__weigths) + self.__bias
        Y_predicted = self._sigmoid(linear)
        class_f = [1 if y > 0.5 else 0 for y in Y_predicted]
        class_f = np.array(class_f)
        return class_f

    def evaluate(self, X_test, Y_test):
        res = self.predict(X_test)
        acc = 0
        for i in range(len(res)):
            if res[i] == Y_test[i]:
                acc += 1
        return acc / len(Y_test)

    def _error(self, H, Y, L1=0):
        print((L1 / (2 * len(H))) * np.sum(np.abs(self.__weigths)))
        er = - np.mean(((Y * np.log(H + 1e-10)) + ((1 - Y) * np.log(1 - H + 1e-10))) + (L1 / (2 * len(H))) * np.sum(
            np.abs(self.__weigths)))
        return er

    def _rand_weights(self, size, type="rand"):
        if type == "rand":
            return np.random.uniform(-1, 1, size=size)
        else:
            return np.zeros(size)

    def _sigmoid(self, x):
        X = [(1 / (1 + np.exp(-z))) for z in x]
        return np.array(X)

    def get_weights(self):
        return self.__weigths

    def set_weights(self, weigths):
        self.__weigths = weigths
