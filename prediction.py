import numpy as np
from sklearn.metrics import mean_squared_error

def prediction(model,x_test):
    preds = model.predict(x_test)
    preds = np.array(preds)
    preds = preds.reshape((6, 587))
    return preds

def evaluate(y_preds,y_true):
    l1=[]
    y_true = y_true.to_numpy()
    for i in range(0,6):
        error = mean_squared_error(y_true[:,i],y_preds[i])
        error = error**0.5
        l1.append(error)
    # return np.mean(l1)
    return [l1, np.mean(l1)]