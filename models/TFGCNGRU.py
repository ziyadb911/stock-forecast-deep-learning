import tensorflow as tf
import spektral as st


class TFGCNGRU:
    def __init__(self, n_classes=1):
        # Get model hyperparameters
        self.n_classes = n_classes

    def build_model(self,
                    x_input_shape,
                    g_input_shape,
                    dropout=0.25,
                    unit=128
                    ):
        self.dropout = dropout

        x_inputs = tf.keras.Input(shape=x_input_shape)
        g_inputs = tf.keras.Input(shape=g_input_shape)
        x = x_inputs
        g = g_inputs
        
        # Graph Convolutional Layer
        x = st.layers.GCNConv(unit, activation='relu')([x, g])
        x = tf.keras.layers.GRU(unit, activation="tanh")(x)
        x = tf.keras.layers.Dropout(self.dropout)(x)

        # output layer
        outputs = tf.keras.layers.Dense(self.n_classes)(x)

        model = tf.keras.Model(inputs=[x_inputs, g_inputs], outputs=outputs)
        
        return model
