"""Builds the MNIST front-forwards network

Implements the inference/loss/training pattern for model building

1. inference() - Builds the model as far as required for running the network forward to make predictions.
2. loss() - Adds to the inference model the layers required to generate loss.
3. training() - Adds to the loss model the Ops required to generate and apply gradients.

This file is used by the various "mnist_*.py" files and not meant to be run.

"""

import math
import tensorflow as tf

# The MNIST dataset has 10 classes, representing the digits 0 through 9.
NUM_CLASSES = 10

# The MNIST images are always 28x28 pixels.
IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE


def weight_variable(shape, stddev):
    initial = tf.truncated_normal(shape=shape, stddev=stddev)
    return tf.Variable(initial, name="weights")


def bias_variable(shape):
    return tf.Variable(tf.zeros(shape), name="biases")


def inference(images, hidden1_units, hidden2_units):
    """Build the MNIST model up to where it may be used for inference.

    Args:
        images: Images placeholder, from inputs().
        hidden1_units: Size of the first hidden layer.
        hidden2_units: Size of the second hidden layer.

    Returns:
        softmax_linear: Output tensor with the computed logits
    """

    # Hidden 1
    with tf.name_scope("hidden1"):
        weights = weight_variable(
            shape=[IMAGE_PIXELS, hidden1_units],
            stddev=1.0 / math.sqrt(float(IMAGE_PIXELS)))

        biases = bias_variable(shape=[hidden1_units])

        hidden1 = tf.nn.relu(tf.matmul(images, weights) + biases)

    # Hidden 2
    with tf.name_scope("hidden2"):
        weights = weight_variable(
            shape=[hidden1_units, hidden2_units],
            stddev=1.0 / math.sqrt(float(hidden1_units)))

        biases = bias_variable(shape=[hidden2_units])

        hidden2 = tf.nn.relu(tf.matmul(hidden1, weights) + biases)

    # Linear
    with tf.name_scope("softmax_linear"):
        weights = weight_variable(
            shape=[hidden2_units, NUM_CLASSES],
            stddev=1.0 / math.sqrt(float(hidden2_units)))

        biases = bias_variable(shape=[NUM_CLASSES])

        logits = tf.matmul(hidden2, weights) + biases

    return logits


def loss(logits, labels):
    """Calculates the loss from the logits and the labels


    Args:
        logits: logits tensor, float - [batch_size, NUM_CLASSES].
        labels: labels tensor, int32 - [batch_size]

    Returns:
        loss: Loss tensor of type float.

    """
    labels = tf.to_int64(labels)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels, logits=logits, name="xentropy")
    loss = tf.reduce_mean(cross_entropy, name="xentropy_mean")
    return loss


def training(loss, learning_rate):
    """Sets up the training Ops.

    Creates a summarizer to track the loss over time in TensorBoard.

    Creates an optimizer and applies the gradients to all trainable variables.

    The Op returned by this function is what must be passed to the `sess.run()` call to cause the model to train.

    Args:
        loss: Loss tensor, from loss().
        learning_rate: The learning rate to use for gradient descent.

    Returns:
        train_op: The Op for training.
    """

    # Add a scalar summary for the snapshot loss.
    tf.summary.scalar("loss", loss)

    # Create the gradient descent optimizer with the given learning rate.
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    # Create a variable to track the global step.
    global_step = tf.Variable(0, name="global_step", trainable=False)
    # Use the optimizer to apply the gradients that minimize the loss
    # (and also increment the global step counter) as a single training step.
    train_op = optimizer.minimize(loss, global_step=global_step)

    return train_op


def evaluation(logits, labels):
    """Evaluate the quality of the logits at predicting the label.

    Args:
        logits: Logits tensor, float - [batch_size, NUM_CLASSES].
        labels: labels tensor, int32 - [batch_size], with values in the range [0, NUM_CLASSES).

    Returns:
        A scalar int32 tensor with the number of examples (out of batch_size)
        that were predicted correctly.
    """

    # For a classifier model, we can use the in_top_k op.
    # It returns a bool tensor with shape [batch_size] that is true for
    # the examples where the label is the top k (here k=1)
    # of all logits for the example.
    targets=[tf.cast(l,tf.int32) for l in logits]
    correct = tf.nn.in_top_k(targets, labels, 1)
    # return the number of true entries
    return tf.reduce_sum(tf.cast(correct, tf.int32))