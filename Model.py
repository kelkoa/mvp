import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import backend as K
from tensorflow_probability import distributions as ds
from tensorflow.keras import initializers, regularizers
import numpy as np

class Attention(layers.Layer):
    def __init__(self, units):
        super(Attention, self).__init__()
        self.W1 = layers.Dense(units)
        self.W2 = layers.Dense(units)
        self.V = layers.Dense(1)

    def call(self, full, last):
        last_extended = tf.expand_dims(last, 1)
        score = self.V(K.tanh(self.W1(full) + self.W2(last_extended)))
        attention_weights = K.softmax(score, axis=1)
        context_vector = attention_weights * full
        context_vector = K.sum(context_vector, axis=1)
        return context_vector

class MVP(layers.Layer):
    def __init__(self):
        super(MVP, self).__init__()
        self.grup = layers.GRU(64, return_sequences=True, return_state=True)
        self.attnp = Attention(64)
        self.tweet_gru = layers.GRU(64, return_sequences=True, return_state=True)
        self.attn_tweet = Attention(64)
        self.grut = layers.GRU(64, return_sequences=True, return_state=True)
        self.attnt = Attention(64)
        self.concat = layers.Concatenate()

        self.linear_h = layers.Dense(32)
        self.linear_mean = layers.Dense(1)
        self.linear_stddev = layers.Dense(1)

        self.concat_out = layers.Concatenate()
        self.linear_final = layers.Dense(2)
        self.softmax = layers.Softmax()

    def call(self, inputs):
        prices = inputs[0]
        tweets = inputs[1]
        labels = inputs[2][:, 0]
        labels_binary = inputs[2][:, 1:]

        seq_x, x = self.grup(prices)
        x_att = self.attnp(seq_x, x)

        seq_y, y = self.tweet_gru(K.reshape(tweets, [-1, tweets.shape[2], tweets.shape[3]]))
        y_att = self.attn_tweet(seq_y, y)

        seq_text, text = self.grut(K.reshape(y_att, [-1, tweets.shape[1], 64]))
        text_att = self.attnt(seq_text, text)
        combined = self.concat([x_att, text_att])
        out_h = self.linear_h(combined)

        out_mean = K.squeeze(self.linear_mean(out_h), axis=-1)
        out_std = K.squeeze(self.linear_stddev(out_h), axis=-1)
        out_std = tf.sqrt(tf.exp(out_std))

        nll_loss = K.log(out_std**2) + (labels - out_mean)**2 / out_std**2

        x_mean = K.mean(prices[:, 0], axis=1)
        x_stddev = K.std(prices[:, 0], axis=1)
        pdf_x = ds.Normal(loc=x_mean, scale=x_stddev)

        pdf_out = ds.Normal(loc=out_mean, scale=out_std)
        kl_loss = ds.kl_divergence(pdf_x, pdf_out)

        max_sharpe = tf.divide(out_mean, tf.square(out_std))
        mvir = max_sharpe * labels

        total_loss = kl_loss + 1e-1 * nll_loss
        self.add_metric(total_loss, name='loss')
        self.add_metric(mvir, name='mvir')
        self.add_loss(total_loss)

        return [out_mean, out_std]
