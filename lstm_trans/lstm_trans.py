import os
import tensorflow as tf
import time
import matplotlib.pyplot as plt
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization, Dropout
from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

class AddAndNorm(layers.Layer):
    def __init__(self, epsilon=1e-6):
        super().__init__()
        self.epsilon = epsilon
        self.norm = layers.LayerNormalization(epsilon=self.epsilon)
        self.feed_forward = layers.Dense(units=128, activation='relu')  # Add a dense layer for feed forward
        
    def call(self, inputs, sublayer):
        # Add the inputs to the outputs of the sublayer
        add = layers.Add()([inputs, sublayer])
        # Apply the feedforward layer
        ff = self.feed_forward(add)
        # Normalize the outputs using layer normalization
        norm = self.norm(ff)
        return norm

def lstm_trans(x_train, y_train, x_valid, y_valid, base_path, model_path):
    start_time = time.time()
    
    text_vector = TextVectorization(max_tokens=37000)
    text_vector.adapt(x_train.values)
    embed = layers.Embedding(input_dim=37000,output_dim=128,mask_zero=True)
    inputs = layers.Input(shape=(1,),dtype=tf.string)
    x = text_vector(inputs)
    x = embed(x)

    x = layers.LSTM(128,return_sequences=True)(x)
    x = layers.LSTM(64)(x)
    x = layers.Reshape((-1, 64))(x)

    transformer = MultiHeadAttention(num_heads=8, key_dim=64)
    #x = transformer(x, x)
    #x = LayerNormalization(epsilon=1e-6)(x)
    x = AddAndNorm()(x, transformer(x, x))
    x = Dropout(0.1)(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(128,activation='relu')(x)
    x = layers.Dropout(0.25)(x)
    x = layers.Dense(64,activation='relu')(x)
    x = layers.Dropout(0.4)(x)


    output_1 = layers.Dense(1,activation='relu',name='o1')(x)
    output_2 = layers.Dense(1,activation='relu',name='o2')(x)
    output_3 = layers.Dense(1,activation='relu',name='o3')(x)
    output_4 = layers.Dense(1,activation='relu',name='o4')(x)
    output_5 = layers.Dense(1,activation='relu',name='o5')(x)
    output_6 = layers.Dense(1,activation='relu',name='o6')(x)

    model_2 = tf.keras.Model(inputs=inputs,outputs=[output_1,output_2,output_3,output_4,output_5,output_6])
    model_2.summary()
    
    # transformer with lstm
    model_2.compile(loss={'o1':tf.keras.losses.MeanSquaredError(),
                        'o2':tf.keras.losses.MeanSquaredError(),
                        'o3':tf.keras.losses.MeanSquaredError(),
                        'o4':tf.keras.losses.MeanSquaredError(),
                        'o5':tf.keras.losses.MeanSquaredError(),
                        'o6':tf.keras.losses.MeanSquaredError()},
                    optimizer='adam',
                    metrics=[tf.keras.metrics.RootMeanSquaredError()])
    # parameters
    checkpoint = ModelCheckpoint(model_path,
                                    monitor="val_loss",
                                    mode="min",
                                    save_best_only = True,
                                    verbose=1)

    earlystop = EarlyStopping(monitor = 'val_loss', 
                                    min_delta = 0, 
                                    patience = 5,
                                    verbose = 1,
                                    restore_best_weights = True)

    learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss', 
                                                    patience=3, 
                                                    verbose=1, 
                                                    factor=0.2, 
                                                    min_lr=0.00000001)
    
    ## fitting the model
    valid_data = (x_valid,y_valid)
    history = model_2.fit(x=x_train,y=y_train,batch_size=32,epochs=25,validation_data=valid_data,callbacks=[earlystop, checkpoint, learning_rate_reduction])
    end_time = time.time()
    training_time_per_epoch = (end_time - start_time) / len(history.history['loss'])
    
    hist = history.history
    loss_keys = ['o1_loss', 'o2_loss', 'o3_loss', 'o4_loss', 'o5_loss', 'o6_loss']
    val_loss_keys = ['val_o1_loss', 'val_o2_loss', 'val_o3_loss', 'val_o4_loss', 'val_o5_loss', 'val_o6_loss']
    rmse_keys = ['o1_root_mean_squared_error', 'o2_root_mean_squared_error', 'o3_root_mean_squared_error', 'o4_root_mean_squared_error', 'o5_root_mean_squared_error', 'o6_root_mean_squared_error']
    val_rmse_keys = ['val_o1_root_mean_squared_error', 'val_o2_root_mean_squared_error', 'val_o3_root_mean_squared_error', 'val_o4_root_mean_squared_error', 'val_o5_root_mean_squared_error', 'val_o6_root_mean_squared_error']

    epochs_range = range(1, len(hist['loss']) + 1)

    plt.figure(figsize=(15,5))
    plt.suptitle('LSTM - Transformer')
    # Plot the RMSE history
    plt.subplot(1, 2, 1)
    for i in range(len(rmse_keys)):
        plt.plot(epochs_range, hist[rmse_keys[i]], label=f'{rmse_keys[i][:]}')
    plt.legend(loc="best")
    plt.xlabel('Epochs')
    plt.ylabel('Train RMSE')
    plt.title('Model Train RMSE')

    # Plot the loss history
    plt.subplot(1, 2, 2)
    for i in range(len(loss_keys)):
        plt.plot(epochs_range, hist[loss_keys[i]], label=f'{loss_keys[i][:]}')
    plt.legend(loc="best")
    plt.xlabel('Epochs')
    plt.ylabel('Train Loss')
    plt.title('Model Train Loss')
    plt.savefig(os.path.join(base_path, 'train_hist.jpg'))

    plt.figure(figsize=(15,5))
    plt.suptitle('LSTM - Transformer')
    # Plot the RMSE history
    plt.subplot(1, 2, 1)
    for i in range(len(val_rmse_keys)):
        plt.plot(epochs_range, hist[val_rmse_keys[i]], label=f'{val_rmse_keys[i][:]}')
    plt.legend(loc="best")
    plt.xlabel('Epochs')
    plt.ylabel('Validation RMSE')
    plt.title('Model Valiadtion RMSE')

    # Plot the loss history
    plt.subplot(1, 2, 2)
    for i in range(len(val_loss_keys)):
        plt.plot(epochs_range, hist[val_loss_keys[i]], label=f'{val_loss_keys[i][:]}')
    plt.legend(loc="best")
    plt.xlabel('Epochs')
    plt.ylabel('Validation Loss')
    plt.title('Model Validation Loss')    
    
    plt.savefig(os.path.join(base_path, 'val_hist.jpg'))
    
    return model_2, training_time_per_epoch