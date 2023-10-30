import tensorflow as tf


class TFGRULSTM:
    def __init__(self, n_classes=1):
        # Get model hyperparameters
        self.n_classes = n_classes

    def build_model(self,
                    input_shape,
                    dropout=0.25,
                    unit=128
                    ):
        self.dropout = dropout

        inputs = tf.keras.Input(shape=input_shape)
        x = inputs
        x = tf.keras.layers.GRU(unit, activation="tanh", return_sequences=True)(x)
        x = tf.keras.layers.LSTM(unit, activation="tanh")(x)
        x = tf.keras.layers.Dropout(self.dropout)(x)

        # output layer
        outputs = tf.keras.layers.Dense(self.n_classes)(x)

        return tf.keras.Model(inputs, outputs)
