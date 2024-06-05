import numpy as np

# Defining the cost function
def my_cost_fn(y_train, y_preds, rp):
  # Compute squared error
  squared_error =  np.square(np.subtract(y_preds, y_train))
  
  weights_squared_sum = np.sum(np.square(model.weights_2d.squeeze()))
  
  regularization = (rp * weights_squared_sum)/(2*len(y_train))

  # Compute average squared error
  avg_squared_error = np.mean(squared_error) / 2
    
    # Add regularization to the cost
  cost = avg_squared_error + regularization
    
  return cost

reg_params = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1] # These are the different values of regularization parameter values

for rp in reg_params:
    print("Costs with regularization parameter = ", rp)
    print("The cost for the training set is: ", my_cost_fn(y_train.squeeze(), y_preds_train, rp))
    print("The cost for the cross validation set is: ", my_cost_fn(y_cv, y_preds_cv, rp))
    print("The cost for the test set is: ", my_cost_fn(y_test, y_preds_test, rp))
    print("\n")

# defining the derivatives.
def derivatives(y_preds, y_train, x_train,rp):
  d={"d_w":[], "d_b":[]}
  dJ_dw_list=[]
  for j in range(len(model.weights_2d.squeeze())):
    regularization = (rp*np.sum(model.weights_2d.squeeze()))/(len(y_train))
    for i in range(x_train.shape[0]):
        dJ_dw_list.append((y_preds[i] - y_train[i])*x_train[i][j])
    d["d_w"].append(np.mean(np.array(dJ_dw_list))+regularization)
    dJ_dw_list.clear()
  
  dJ_db_list=[]
  for i in range(x_train.shape[0]):
    dJ_db_list.append((y_preds[i] - y_train[i]))
  d["d_b"].append(np.mean(np.array(dJ_db_list)))
  return d

#training the model using Gradient descent Algorithm
epochs = # prefered number of epochs
learning_rate = # your prefered learning rate
    
for epoch in range(epochs):
    d=derivatives(y_preds_train,y_train,x_train_norm,rp)
    for j in range(len(model.weights_2d)):
        model.weights_2d[j] = model.weights_2d[j] - (learning_rate*d["d_w"][j])
    model.bias = model.bias - (learning_rate*d["d_b"][0])

    y_preds_train = model.predict(x_train_norm)
    
    cost_fn = my_cost_fn(y_train.squeeze(), y_preds_train,rp)


    if epoch%30==0:
        print("cost function:",cost_fn)