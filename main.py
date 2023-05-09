import os
import time
from visualisation import visualisation
from lstm_lstm.lstm_lstm import lstm_lstm
from lstm_trans.lstm_trans import lstm_trans
from trans_trans.trans_trans import trans_trans
from trans_lstm.trans_lstm import trans_lstm
from prediction import prediction, evaluate
from tensorflow.keras.models import load_model
def main():
    score = []
    train_time_per_epoch = []
    pred_time = []
    x_train, y_train, x_valid, y_valid, x_test, y_test = visualisation(os.path.curdir)
    ###### lstm-lstm
    base_path = 'lstm_lstm'
    model_path = os.path.join(base_path, 'weight')
    if not os.path.exists(model_path):
        os.mkdir(model_path)
        model, time_temp = lstm_lstm(x_train, y_train, x_valid, y_valid, base_path, model_path)
        train_time_per_epoch.append(time_temp)
    else:
        model = load_model(model_path)
    start_time = time.time()
    score.append(evaluate(prediction(model, x_test), y_test))
    end_time = time.time()
    pred_time.append(end_time-start_time)
    
    print('Score of LSTM - LSTM model is ', evaluate(prediction(model, x_test), y_test))
    ######
    
    ###### lstm - trans
    base_path = 'lstm_trans'
    model_path = os.path.join(base_path, 'weight')
    if not os.path.exists(model_path):
        os.mkdir(model_path)
        model, time_temp = lstm_trans(x_train, y_train, x_valid, y_valid, base_path, model_path)
        train_time_per_epoch.append(time_temp)
    else:
        model = load_model(model_path)
    start_time = time.time()
    score.append(evaluate(prediction(model, x_test), y_test))
    end_time = time.time()
    pred_time.append(end_time-start_time)
    print('Score of LSTM - Transformer model is ', evaluate(prediction(model, x_test), y_test))
    ######
    
    ###### trans - trans
    base_path = 'trans_trans'
    model_path = os.path.join('trans_trans', 'weight')
    if not os.path.exists(model_path):
        os.mkdir(model_path)
        model, time_temp = trans_trans(x_train, y_train, x_valid, y_valid, base_path, model_path)
        train_time_per_epoch.append(time_temp)
    else:
        model = load_model(model_path)
    start_time = time.time()
    score.append(evaluate(prediction(model, x_test), y_test))
    end_time = time.time()
    pred_time.append(end_time-start_time)
    print('Score of Transformer - Transformer model is ', evaluate(prediction(model, x_test), y_test))
    ######
    
    ###### trans - lstm
    base_path = 'trans_lstm'
    model_path = os.path.join('trans_lstm', 'weight')
    if not os.path.exists(model_path):
        os.mkdir(model_path)
        model, time_temp = trans_lstm(x_train, y_train, x_valid, y_valid, base_path, model_path)
        train_time_per_epoch.append(time_temp)
    else:
        model = load_model(model_path)
    start_time = time.time()
    score.append(evaluate(prediction(model, x_test), y_test))
    end_time = time.time()
    pred_time.append(end_time-start_time)
    print('Score of Transformer - LSTM model is ', evaluate(prediction(model, x_test), y_test))
    ######
    print(score)
    print(train_time_per_epoch)
    print(pred_time)
main()