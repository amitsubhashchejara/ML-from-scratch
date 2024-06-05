import numpy as np

class Linear_regression():
  def __init__(self):
    self.weights = np.zeros(5)
    self.weights_2d = np.expand_dims(self.weights, axis=1)
    self.bias = np.zeros(1)

  def predict(self,x):
    y_preds_2d = np.matmul(self.weights_2d.T,x.T)+self.bias
    y_preds = np.squeeze(y_preds_2d)

    return y_preds

# Getting the initial model parameters
model = Linear_regression()
print("The initial weights are: ", model.weights_2d)
print("The initial bias is: ", model.bias)
print("The shape of the initial weights is: ", model.weights_2d.shape)
print("The shape of input is: ", x_train_norm.shape)  
