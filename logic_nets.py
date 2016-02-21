import tensorflow as tf
import numpy as np

# binary logic functions that support per-element operations between numpy
# nd arrays.
xor_fn = np.vectorize(lambda x, y: x != y)
nand_fn = np.vectorize(lambda x, y: not (x and y))
and_fn = np.vectorize(lambda x, y: x and y)
or_fn = np.vectorize(lambda x, y: x or y)

def sample_logic_fn(f, samples=10000):
    """ Takes a vectorized function that takes two binary input values and
    generates labeled data for use in training or evaluating classifiers.
    """
    inputs = np.random.randint(2, size=(samples, 2))
    outputs = f(inputs[:,0], inputs[:,1])
    labels = np.transpose(np.vstack((outputs, outputs==False)))
    return inputs.astype(np.float32), labels.astype(np.float32)


def nn_model(n_input, n_classes, hiddens=[]):
    """ Construct a layer map that consists of a list of TensorFlow variables
    (Tensors) representing the weights and biases from each layer to the next
    layer defined by the input data, the number of output labels (output layer
    corresponds to one-hot vector encodings for classes), and an ordered count
    of neurons in hidden layers.
    """
    layer_map = []
    connections = [n_input] + hiddens + [n_classes]
    for n1, n2 in zip(connections[:-1], connections[1:]):
        weights = tf.Variable(tf.truncated_normal([n1, n2]))
        biases = tf.Variable(tf.zeros([n2]))
        layer_map.append((weights, biases))
    return layer_map


def feed_forward(layer_map, data):
    """ Use a layer_map returned by nn_model and use a reduce-like composition
    model to construct the TensorFlow data flow graph that specifies the matrix
    multiplications that calculates the response for each neuron in each layer
    from the response of all connected neurons in the previous layer plus the
    biases and with the rectified linear activation function.
    """
    accum = data
    for l in layer_map[:-1]:
        accum = tf.nn.relu(tf.matmul(accum, l[0]) + l[1])
    return tf.matmul(accum, layer_map[-1][0]) + layer_map[-1][1]


def accuracy(preds, labels):
    """ Return the mean error of the difference between one hot encoded
    labels and softmax neural network output probabilities.
    """
    return np.mean(np.argmax(preds,1) == np.argmax(labels,1))


def fit_nnet(train_data, train_labels, hiddens=[], n_steps=1000,
             lrate=0.01, valid_data=None, valid_labels=None, test_data=None,
             test_labels=None):
    """ Construct a neural net data flow graph in TensorFlow and use it to learn
    weights that fit the train_data to the train_labels.
    """

    # matrix dimension influencing variables
    n_samples, input_dims = np.shape(train_data)
    n_labels = np.shape(train_labels)[-1]

    # build up TensorFlow data flow graph by initializing variables and
    # placeholders and composing functions.
    graph = tf.Graph()
    with graph.as_default():
        tf_train_data = tf.Variable(train_data)
        tf_train_labels = tf.Variable(train_labels)

        # build layer stack then build tf dataflow graph implied by feed_forward
        model = nn_model(input_dims, n_labels, hiddens=hiddens)
        pred = feed_forward(model, tf_train_data)

        # use log distance of labels from predictions as our cost
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(pred, tf_train_labels)
        cost_fn = tf.reduce_mean(cross_entropy)

        # use gradient descent as our optimizer
        opt = tf.train.GradientDescentOptimizer(lrate).minimize(cost_fn)

        # determine predictions for training data
        train_pred = tf.nn.softmax(pred)

        # set up variables for and calculate predictions for validation data
        # and test data if provided
        if valid_data is not None:
            tf_valid_data = tf.constant(valid_data)
            tf_test_data = tf.constant(test_data)
        if test_data is not None:
            valid_pred = tf.nn.softmax(feed_forward(model, tf_valid_data))
            test_pred = tf.nn.softmax(feed_forward(model, tf_test_data))

    with tf.Session(graph=graph) as session:
        tf.initialize_all_variables().run()
        for step in range(n_steps):
            _, cost_y, preds = session.run([opt, cost_fn, train_pred])
            if (step % 200) == 0:
                print("Cost function value as of step %d: %f"\
                    % (step, cost_y))
                print("Accuracy for training dataset: %.3f"\
                    % accuracy(preds, train_labels))
                if valid_data is not None:
                    print("Validation accuracy estimate: %.3f"\
                        % accuracy(valid_pred.eval(), valid_labels))
        if test_data is not None:
            print("Test accuracy after training complete: %.3f"\
                % accuracy(test_pred.eval(), test_labels))


if __name__ == "__main__":
    # we can get 100% on any of these with just a bit of training
    for l_fn in [and_fn, or_fn, nand_fn, xor_fn]:
        train_data, train_labels = sample_logic_fn(l_fn)
        valid_data, valid_labels = sample_logic_fn(l_fn, 1000)
        test_data, test_labels = sample_logic_fn(l_fn, 1000)
        fit_nnet(train_data, train_labels, hiddens=[3], n_steps=2000,
                lrate=0.75, valid_data=valid_data, valid_labels=valid_labels,
                test_data=test_data, test_labels=test_labels)
