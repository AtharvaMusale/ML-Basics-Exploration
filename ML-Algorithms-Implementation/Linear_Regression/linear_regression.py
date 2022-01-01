import numpy as np

class LinearRegression:
  def __init__(self,learning_rate=0.01, iter = 30000 ):
    self.lr = learning_rate
    self.num_iter = iter
   
  def y_hat(self,X,w):
    y_hat =  np.dot(w.T,X)
    return y_hat
  
  def loss(self, yhat, y):
    L = 1/self.m * np.sum(np.power(y - yhat,2))
    return L

  def gradient_descent(self, w, X , y, yhat):
    dldw = 2/self.m * np.dot(X, (yhat-y).T)
    w = w - self.lr * dldw
    return w
  
  def main(self,X,y,loss_list):
    ones = np.ones((1,X.shape[1]))
    X = np.append(ones,X,axis=0)

    self.m = X.shape[1]
    self.n = X.shape[0]

    w = np.zeros((self.n,1))
    for i in range(self.num_iter + 1):
      yhat = self.y_hat(X,w)
      cost = self.loss(yhat, y )
      if i % 10000 == 0:
        print(f'Loss at iteration {i} is {cost}')
        loss_list.append(cost)
      w = self.gradient_descent(w, X, y, yhat)

    return w

if __name__ == '__main__':
  loss_list = []
  X = np.random.rand(1, 500)
  y = 3 * X + 5 + np.random.randn(1, 500) * 0.1
  regression = LinearRegression()
  w = regression.main(X, y, loss_list)