import numpy as np
import tensorflow as tf

def get_lstm_cell(cell_size, output_keep_prob):
    """
    Get LSTM cell with given size and given dropout
    """
    cell = tf.contrib.rnn.LSTMCell(cell_size, state_is_tuple=True)
    cell = tf.contrib.rnn.DropoutWrapper(cell=cell, output_keep_prob=output_keep_prob)
    return cell


def get_multicells(n_hidden, keep_prop, n_layers):
    cells = [get_lstm_cell(n_hidden, keep_prop) for _ in range(n_layers)]
    return tf.contrib.rnn.MultiRNNCell(cells)

class Model():
    pass


def get_model(n_features, n_classes, n_layers, n_hidden):
    input_placeholder = tf.placeholder(tf.float32, [None, None, n_features])
    output_placeholder = tf.placeholder(tf.float32, [None, n_classes])
    output_keep_prob = tf.placeholder(tf.float32)
    seq_len = tf.placeholder(tf.int32)
    learning_rate = tf.placeholder(tf.float32)
    W_fc1 = tf.Variable(tf.truncated_normal([4*n_hidden, 4*n_hidden])) 
    b_fc1 = tf.Variable(tf.constant(0.1, shape=[4*n_hidden]))
    W = tf.Variable(tf.truncated_normal([4*n_hidden, n_classes]))
    b = tf.Variable(tf.constant(0.1, shape=[n_classes]))

    fw_cells = get_multicells(n_hidden, output_keep_prob, n_layers)
    bw_cells = get_multicells(n_hidden, output_keep_prob, n_layers)

    outputs, _ = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw=fw_cells,
                    cell_bw=bw_cells,
                    inputs=input_placeholder,
                    sequence_length=seq_len,
                    dtype=tf.float32)
    
    concatenated_outputs = tf.concat(outputs, axis=-1)

    # Gather last relevant timestep as dictated by the sequence length
    ind = tf.convert_to_tensor(seq_len) - 1
    batch_range = tf.range(tf.shape(input_placeholder)[0])
    indices = tf.stack([batch_range, ind], axis=1)
    last_states = tf.gather_nd(concatenated_outputs, indices)
    conc_last_states = tf.reshape(last_states, [-1, 4*n_hidden])

    fc1 = tf.nn.sigmoid(tf.matmul(conc_last_states, W_fc1)+b_fc1)
    logits_ = tf.nn.sigmoid(tf.matmul(fc1, W)+b)
    cross_ent = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits_, labels=output_placeholder)
    mean_error = tf.reduce_mean(cross_ent)
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(mean_error)

    model = Model()
    model.input = input_placeholder
    model.output = output_placeholder
    model.keep_prob = output_keep_prob
    model.seq_len = seq_len
    model.learning_rate = learning_rate
    model.logits = logits
    model.cross_ent = cross_ent
    model.mean_error = mean_error
    model.train_op = train_op

    return model