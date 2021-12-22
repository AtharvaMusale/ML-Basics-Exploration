import numpy as np
from sklearn.datasets import make_blobs
class LogisticRegression:

  def __init__(self, X, learning_rate=0.1, num_iter=10000):
    self.lr = learning_rate
    self.num_iter = num_iter
    self.m , self.n = X.shape


  def train(self, X, y):
      self.weights = np.zeros((self.n, 1))
      self.bias = 0
      for it in range(self.num_iter + 1):
          y_predict = self.sigmoid(np.dot(X, self.weights) + self.bias)
          cost = (
              -1
              / self.m
              * np.sum(y * np.log(y_predict) + (1 - y) * np.log(1 - y_predict))
          )
          dw = 1 / self.m * np.dot(X.T, (y_predict - y))
          db = 1 / self.m * np.sum(y_predict - y)
          self.weights -= self.lr * dw
          self.bias -= self.lr * db
          if it % 1000 == 0:
              print(f"Cost after iteration {it}: {cost}")

      return self.weights, self.bias



  def predict(self,X):
    y_predict = self.sigmoid(np.dot(X,self.weights)+self.bias)
    y_predict_labels = y_predict>0.5
    return y_predict_labels


  def sigmoid(self,z):
    return (1/(1+np.exp(-z)))

  
if __name__ == '__main__':
    np.random.seed(1)
    X, y = make_blobs(n_samples=100000, centers=2)
    y = y[:, np.newaxis]

    logreg = LogisticRegression(X)
    w, b = logreg.train(X, y)
    y_predict = logreg.predict(X)

    print(f"Accuracy: {np.sum(y==y_predict)/X.shape[0]}")