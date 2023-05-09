import os
import tensorflow as tf
import numpy as np
import time
import matplotlib.pyplot as plt
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization, Dropout
from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau


def positional_encoding(length, depth):
    depth = depth/2

    positions = np.arange(length)[:, np.newaxis]     # (seq, 1)
    depths = np.arange(depth)[np.newaxis, :]/depth   # (1, depth)

    angle_rates = 1 / (10000**depths)         # (1, depth)
    angle_rads = positions * angle_rates      # (pos, depth)

    pos_encoding = np.concatenate(
        [np.sin(angle_rads), np.cos(angle_rads)],
        axis=-1) 

    return tf.cast(pos_encoding, dtype=tf.float32)

class PositionalEmbedding(tf.keras.layers.Layer):
  def __init__(self, vocab_size, d_model):
    super().__init__()
    self.d_model = d_model
    self.embedding = tf.keras.layers.Embedding(vocab_size, d_model, mask_zero=True) 
    self.pos_encoding = positional_encoding(length=2048, depth=d_model)

  def compute_mask(self, *args, **kwargs):
    return self.embedding.compute_mask(*args, **kwargs)

  def call(self, x):
    length = tf.shape(x)[1]
    x = self.embedding(x)
    # This factor sets the relative scale of the embedding and positonal_encoding.
    x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
    x = x + self.pos_encoding[tf.newaxis, :length, :]
    return x

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


def trans_trans(x_train, y_train, x_valid, y_valid, base_path, model_path):
    start_time = time.time()
    # Define the TextVectorization layer
    text_vector = TextVectorization(max_tokens=37000)
    text_vector.adapt(x_train.values)

    # Define the Embedding layer
    #embed = layers.Embedding(input_dim=37000, output_dim=128, mask_zero=True)
    embed = PositionalEmbedding(vocab_size=37000, d_model=128)
    # Define the inputs
    inputs = layers.Input(shape=(1,), dtype=tf.string)

    # Apply the TextVectorization layer
    x = text_vector(inputs)

    # Apply the Embedding layer
    x = embed(x)

    # Apply the positional encoding
    #x = positional_encoding(10000, 128)(x)

    # Apply the transformer layers
    transformer1 = MultiHeadAttention(num_heads=8, key_dim=64)
    #x = transformer1(x, x)
    #x = LayerNormalization(epsilon=1e-6)(x)
    x = AddAndNorm()(x, transformer1(x, x))
    x = Dropout(0.1)(x)

    transformer2 = MultiHeadAttention(num_heads=8, key_dim=64)
    #x = transformer2(x, x)
    #x = LayerNormalization(epsilon=1e-6)(x)
    x = AddAndNorm()(x, transformer2(x, x))
    x = Dropout(0.1)(x)

    # Apply the final layers
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = Dropout(0.25)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = Dropout(0.4)(x)

    output_1 = layers.Dense(1, activation='relu', name='o1')(x)
    output_2 = layers.Dense(1, activation='relu', name='o2')(x)
    output_3 = layers.Dense(1, activation='relu', name='o3')(x)
    output_4 = layers.Dense(1, activation='relu', name='o4')(x)
    output_5 = layers.Dense(1, activation='relu', name='o5')(x)
    output_6 = layers.Dense(1, activation='relu', name='o6')(x)

    # Define the model
    model_3 = tf.keras.Model(inputs=inputs, outputs=[output_1, output_2, output_3, output_4, output_5, output_6])

    # Print the model summary
    model_3.summary()
    
    # transformer
    model_3.compile(loss={'o1':tf.keras.losses.MeanSquaredError(),
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
    history = model_3.fit(x=x_train,y=y_train,batch_size=32,epochs=25,validation_data=valid_data,callbacks=[earlystop, checkpoint, learning_rate_reduction])
    end_time = time.time()
    
    training_time_per_epoch = (end_time - start_time) / len(history.history['loss'])
    
    hist = history.history
    loss_keys = ['o1_loss', 'o2_loss', 'o3_loss', 'o4_loss', 'o5_loss', 'o6_loss']
    val_loss_keys = ['val_o1_loss', 'val_o2_loss', 'val_o3_loss', 'val_o4_loss', 'val_o5_loss', 'val_o6_loss']
    rmse_keys = ['o1_root_mean_squared_error', 'o2_root_mean_squared_error', 'o3_root_mean_squared_error', 'o4_root_mean_squared_error', 'o5_root_mean_squared_error', 'o6_root_mean_squared_error']
    val_rmse_keys = ['val_o1_root_mean_squared_error', 'val_o2_root_mean_squared_error', 'val_o3_root_mean_squared_error', 'val_o4_root_mean_squared_error', 'val_o5_root_mean_squared_error', 'val_o6_root_mean_squared_error']

    epochs_range = range(1, len(hist['loss']) + 1)

    plt.figure(figsize=(15,5))
    plt.suptitle('Transformer - Transformer')
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
    plt.suptitle('Transformer - Transformer')
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
    
    return model_3, training_time_per_epoch