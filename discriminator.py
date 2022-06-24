import tensorflow as tf
import numpy as np


class Discriminator(object):
    """
    Bộ phân loại sử  dụng CNN
    Sử dungh một tầng embedding, một convolutional, max-pooling and softmax layer.
    """

    def __init__(self, sequence_length, num_classes, vocab_size,
            embedding_size, filter_sizes, num_filters, pretrained_path=False, l2_reg_lambda=0.0):

        # Placeholders cho  input, output và dropout
        self.input_x = tf.compat.v1.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.compat.v1.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.compat.v1.placeholder(tf.float32, name="dropout_keep_prob")
        self.pretrained_path = pretrained_path

        # L_2 loss
        l2_loss = tf.constant(0.0)

        with tf.variable_scope('discriminator'):

            # Tầng Embedding 
            with tf.name_scope("embedding"):
                self.embedding_matrix = tf.Variable(tf.random_normal([vocab_size, embedding_size]), name="embedding_matrix")
                
                self.embedded_chars = tf.nn.embedding_lookup(self.embedding_matrix, self.input_x)
                self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

            # Tạo một (convolution +   max pooling) cho mỗi filter size
            pooled_outputs = []
            for filter_size, num_filter in zip(filter_sizes, num_filters):
                with tf.name_scope("conv-maxpool-%s" % filter_size):

                    # tầng Convolution
                    filter_shape = [filter_size, embedding_size, 1, num_filter]
                    W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                    b = tf.Variable(tf.constant(0.1, shape=[num_filter]), name="b")
                    conv = tf.nn.conv2d(
                        self.embedded_chars_expanded,
                        W,
                        strides=[1, 1, 1, 1],
                        padding="VALID",
                        name="conv")

                    # Áp dụng non-linearity
                    h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")

                    # Maxpooling trên mỗi outputs
                    pooled = tf.nn.max_pool2d(
                        h,
                        ksize=[1, sequence_length - filter_size + 1, 1, 1],
                        strides=[1, 1, 1, 1],
                        padding='VALID',
                        name="pool")
                    pooled_outputs.append(pooled)

            # So sánh tất cả the pooled features
            num_filters_total = sum(num_filters)
            self.h_pool = tf.concat(pooled_outputs, 3)
            self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

            # Thêm tầng highway
            with tf.name_scope("highway"):
                self.h_highway = highway(self.h_pool_flat, self.h_pool_flat.get_shape()[1], 1, 0)

            # Thêm tầng dropout
            with tf.name_scope("dropout"):
                self.h_drop = tf.nn.dropout(self.h_highway, self.dropout_keep_prob)

            # (không normalized) scores và kết quả dự đoán
            with tf.name_scope("output"):
                W = tf.Variable(tf.truncated_normal([num_filters_total, num_classes], stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
                l2_loss += tf.nn.l2_loss(W)
                l2_loss += tf.nn.l2_loss(b)
                self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
                # Tính softmax
                self.ypred_for_auc = tf.nn.softmax(self.scores)
                self.predictions = tf.argmax(self.scores, 1, name="predictions")

            # Tính trung bình cross-entropy loss
            with tf.name_scope("loss"):
                losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
                self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        self.params = [param for param in tf.trainable_variables() if 'discriminator' in param.name]
        d_optimizer = tf.train.AdamOptimizer(1e-4)
        grads_and_vars = d_optimizer.compute_gradients(self.loss, self.params, aggregation_method=2)
        self.train_op = d_optimizer.apply_gradients(grads_and_vars)

    def init_matrix(self, vocab_size, embedding_size, extend_vocab_size):
        embeddings = []

        embeddings = np.random.random([vocab_size, embedding_size])

        embedding_matrix = np.pad(embeddings,
                                  pad_width=[[4, extend_vocab_size], [0, 0]],
                                  mode='constant',
                                  constant_values=0.).astype(np.float32)
        return embedding_matrix

def linear(input_, output_size, scope=None):

    shape = input_.get_shape().as_list()
    if len(shape) != 2:
        raise ValueError("Linear is expecting 2D arguments: %s" % str(shape))
    if not shape[1]:
        raise ValueError("Linear expects shape[1] of arguments: %s" % str(shape))
    input_size = shape[1]

    # bắt đầu tính tại đây
    with tf.variable_scope(scope or "SimpleLinear"):
        matrix = tf.get_variable("Matrix", [output_size, input_size], dtype=input_.dtype)
        bias_term = tf.get_variable("Bias", [output_size], dtype=input_.dtype)

    return tf.matmul(input_, tf.transpose(matrix)) + bias_term


def highway(input_, size, num_layers=1, bias=-2.0, f=tf.nn.relu, scope='Highway'):
    
    output = None
    with tf.variable_scope(scope):
        for idx in range(num_layers):
            g = f(linear(input_, size, scope='highway_lin_%d' % idx))

            t = tf.sigmoid(linear(input_, size, scope='highway_gate_%d' % idx) + bias)

            output = t * g + (1. - t) * input_
            input_ = output

    return output
