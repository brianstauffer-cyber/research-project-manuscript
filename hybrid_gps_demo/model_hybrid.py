
import tensorflow as tf
from tensorflow.keras import layers, Model

def transformer_block(x, num_heads=4, key_dim=32, ff_dim=128, dropout=0.1, name="xformer"):
    # x: (batch, timesteps, features)
    attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim, name=f"{name}_mha")(x, x)
    x = layers.Add(name=f"{name}_add1")([x, attn])
    x = layers.LayerNormalization(epsilon=1e-6, name=f"{name}_ln1")(x)
    # Feed-forward
    ff = layers.Dense(ff_dim, activation="relu", name=f"{name}_ff1")(x)
    ff = layers.Dropout(dropout, name=f"{name}_drop")(ff)
    ff = layers.Dense(x.shape[-1], name=f"{name}_ff2")(ff)
    x = layers.Add(name=f"{name}_add2")([x, ff])
    x = layers.LayerNormalization(epsilon=1e-6, name=f"{name}_ln2")(x)
    return x

def build_hybrid_model(seq_len=256, n_features=3,
                       conv_filters=64, conv_kernel=5,
                       lstm_units=64, num_heads=4, key_dim=32, ff_dim=128,
                       dropout=0.1):
    inp = layers.Input(shape=(seq_len, n_features))

    # CNN front-end (1D conv over time)
    x = layers.Conv1D(conv_filters, conv_kernel, padding="same", activation="relu")(inp)
    x = layers.Conv1D(conv_filters, conv_kernel, padding="same", activation="relu")(x)
    x = layers.MaxPooling1D(pool_size=2)(x)  # (seq_len/2, filters)

    # LSTM for temporal summarization (return sequences for attention)
    x = layers.LSTM(lstm_units, return_sequences=True)(x)

    # Transformer attention over time
    x = transformer_block(x, num_heads=num_heads, key_dim=key_dim, ff_dim=ff_dim, dropout=dropout)

    # Global pooling + dense head
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    out = layers.Dense(1, activation="sigmoid")(x)

    model = Model(inputs=inp, outputs=out, name="Hybrid_CNN_LSTM_Transformer")
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="binary_crossentropy",
                  metrics=["AUC","Precision","Recall"])
    return model
