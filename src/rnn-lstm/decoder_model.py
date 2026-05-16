import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense, Embedding, SimpleRNN, LSTM, Reshape, Concatenate, RepeatVector

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'utils'))
from text_utils import prepare_training_pair


def build_decoder_model(
    vocab_size: int,
    embed_dim: int,
    hidden_size: int,
    n_layers: int,
    decoder_type: str = "lstm",
    feature_dim: int = 2048,
    max_seq_len: int = 35,
    injection_method: str = "pre",
) -> Model:
    if decoder_type not in ("rnn", "lstm"):
        raise ValueError("decoder_type harus 'rnn' atau 'lstm'")
    if injection_method not in ("pre", "init"):
        raise ValueError("injection_method harus 'pre' atau 'init'")

    image_input   = Input(shape=(feature_dim,),       name="image_features")
    caption_input = Input(shape=(max_seq_len - 1,),   name="caption_tokens", dtype="int32")

    x_img = Dense(embed_dim, name="image_projection")(image_input)
    x_cap = Embedding(vocab_size, embed_dim, name="embedding")(caption_input)

    if injection_method == "pre":
        x_img = Reshape((1, embed_dim), name="image_projection_reshape")(x_img)
        x = Concatenate(axis=1, name="concat_image_caption")([x_img, x_cap])
    else:
        x = x_cap

    for i in range(n_layers):
        if decoder_type == "rnn":
            x = SimpleRNN(hidden_size, return_sequences=True,
                          dropout=0.3, recurrent_dropout=0.1,
                          name=f"simple_rnn_{i}")(x)
        else:
            x = LSTM(hidden_size, return_sequences=True,
                     dropout=0.3, recurrent_dropout=0.1,
                     name=f"lstm_{i}")(x)

    if injection_method == "init":
        x_img_seq = RepeatVector(max_seq_len - 1, name="image_repeat")(x_img)
        x = Concatenate(axis=-1, name="concat_context_image")([x, x_img_seq])

    logits = Dense(vocab_size, activation="softmax", name="output_dense")(x)

    return Model(inputs=[image_input, caption_input], outputs=logits)

@tf.function
def masked_loss(y_true, y_pred, pad_idx: int = 0):
    mask = tf.cast(tf.not_equal(y_true, pad_idx), tf.float32)
    loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
    return tf.reduce_sum(loss * mask) / tf.maximum(tf.reduce_sum(mask), 1.0)


@tf.function
def masked_accuracy(y_true, y_pred, pad_idx: int = 0):
    mask     = tf.cast(tf.not_equal(y_true, pad_idx), tf.float32)
    pred_ids = tf.cast(tf.argmax(y_pred, axis=-1), tf.int32)
    correct  = tf.cast(tf.equal(pred_ids, y_true), tf.float32)
    return tf.reduce_sum(correct * mask) / tf.maximum(tf.reduce_sum(mask), 1.0)


def build_dataset(
    image_names: list,
    all_captions: dict,
    features: dict,
    word2idx: dict,
    max_seq_len: int,
    batch_size: int,
    shuffle: bool = True,
    buffer_size: int = 10000,
    seed: int = 42,
    injection_method: str = "pre",
) -> tf.data.Dataset:
    if injection_method not in ("pre", "init"):
        raise ValueError("injection_method harus 'pre' atau 'init'")
    pad_idx = word2idx["<pad>"]

    img_feats_list  = []
    input_tok_list  = []
    target_los_list = []

    for img_name in image_names:
        feat = features.get(img_name)
        if feat is None:
            continue
        for cap in all_captions.get(img_name, []):
            inp, tgt = prepare_training_pair(cap, word2idx, max_len=max_seq_len)

            if injection_method == "pre":
                target_loss = np.append(tgt, pad_idx).astype(np.int32)
            else:
                target_loss = tgt.astype(np.int32)
            img_feats_list.append(feat)
            input_tok_list.append(inp)
            target_los_list.append(target_loss)

    image_feats  = np.stack(img_feats_list,  axis=0).astype(np.float32)
    input_tokens = np.stack(input_tok_list,  axis=0).astype(np.int32)
    target_loss  = np.stack(target_los_list, axis=0).astype(np.int32)

    ds = tf.data.Dataset.from_tensor_slices(
        ((image_feats, input_tokens), target_loss)
    )
    if shuffle:
        ds = ds.shuffle(buffer_size=buffer_size, seed=seed)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
