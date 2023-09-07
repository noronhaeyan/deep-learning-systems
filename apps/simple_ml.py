import pdb
import struct
import gzip
import numpy as np

import sys
sys.path.append('python/')
import needle as ndl


def parse_mnist(image_filename, label_filename):
    """ Read an images and labels file in MNIST format.  See this page:
    http://yann.lecun.com/exdb/mnist/ for a description of the file format.

    Args:
        image_filename (str): name of gzipped images file in MNIST format
        label_filename (str): name of gzipped labels file in MNIST format

    Returns:
        Tuple (X,y):
            X (numpy.ndarray[np.float32]): 2D numpy array containing the loaded
                data.  The dimensionality of the data should be
                (num_examples x input_dim) where 'input_dim' is the full
                dimension of the data, e.g., since MNIST images are 28x28, it
                will be 784.  Values should be of type np.float32, and the data
                should be normalized to have a minimum value of 0.0 and a
                maximum value of 1.0.

            y (numpy.ndarray[dypte=np.int8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.int8 and
                for MNIST will contain the values 0-9.
    """
    ### BEGIN YOUR SOLUTION
    with gzip.open(image_filename, 'rb') as f:
        magic, size = struct.unpack('>II', f.read(8))
        nrows, ncols = struct.unpack('>II', f.read(8))
        buf = f.read(size * nrows * ncols)
        data = np.frombuffer(buf, dtype=np.dtype(np.uint8)).astype(np.float32)
        # data = np.fromfile(f, dtype = np.dtype(np.uint8), count = size*nrows*ncols)
        X = data.reshape((size, nrows * ncols))
        max_x = np.max(X)
        min_x = np.min(X)
        X = (X - min_x) / (max_x - min_x)

    with gzip.open(label_filename, 'rb') as i:
        magic, size = struct.unpack('>II', i.read(8))
        buf = i.read(size)
        y = np.frombuffer(buf, dtype=np.dtype(np.uint8))

    return X, y
    ### END YOUR SOLUTION


def softmax_loss(Z, y_one_hot):
    """ Return softmax loss.  Note that for the purposes of this assignment,
    you don't need to worry about "nicely" scaling the numerical properties
    of the log-sum-exp computation, but can just compute this directly.

    Args:
        Z (ndl.Tensor[np.float32]): 2D Tensor of shape
            (batch_size, num_classes), containing the logit predictions for
            each class.
        y (ndl.Tensor[np.int8]): 2D Tensor of shape (batch_size, num_classes)
            containing a 1 at the index of the true label of each example and
            zeros elsewhere.

    Returns:
        Average softmax loss over the sample. (ndl.Tensor[np.float32])
    """
    ### BEGIN YOUR SOLUTION

    N = Z.shape[0]

    exp_Z = ndl.exp(Z)

    sum_exp = ndl.summation(exp_Z, axes = (1,))

    log_sum_exp = ndl.log(sum_exp)

    Z_y = ndl.summation(ndl.multiply(Z, y_one_hot), axes = (1,))

    batch_loss = log_sum_exp - Z_y
    #pdb.set_trace()
    return ndl.average(batch_loss)

    ### END YOUR SOLUTION


def nn_epoch(X, y, W1, W2, lr = 0.1, batch=100):
    """ Run a single epoch of SGD for a two-layer neural network defined by the
    weights W1 and W2 (with no bias terms):
        logits = ReLU(X * W1) * W2
    The function should use the step size lr, and the specified batch size (and
    again, without randomizing the order of X).

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        W1 (ndl.Tensor[np.float32]): 2D array of first layer weights, of shape
            (input_dim, hidden_dim)
        W2 (ndl.Tensor[np.float32]): 2D array of second layer weights, of shape
            (hidden_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD mini-batch

    Returns:
        Tuple: (W1, W2)
            W1: ndl.Tensor[np.float32]
            W2: ndl.Tensor[np.float32]
    """

    ### BEGIN YOUR SOLUTION
    #pdb.set_trace()
    N = int(X.shape[0] / batch)

    for i in range(N):
        X_batch = X[i*batch:(i+1)*batch, :]
        y_batch = y[i*batch:(i+1)*batch]

        Z = ndl.relu(ndl.Tensor(X_batch) @ W1) @ W2
        #pdb.set_trace()
        y_one_hot = np.zeros((y_batch.shape[0], W2.shape[-1]))
        y_one_hot[np.arange(y_batch.size), y_batch] = 1
        y_ = ndl.Tensor(y_one_hot)

        loss = softmax_loss(Z, y_)

        loss.backward()

        W1 = (W1 - lr * W1.grad).detach()

        W2 = (W2 - lr * W2.grad).detach()

    #pdb.set_trace()
    return W1, W2

    ### END YOUR SOLUTION


### CODE BELOW IS FOR ILLUSTRATION, YOU DO NOT NEED TO EDIT

def loss_err(h,y):
    """ Helper function to compute both loss and error"""
    y_one_hot = np.zeros((y.shape[0], h.shape[-1]))
    y_one_hot[np.arange(y.size), y] = 1
    y_ = ndl.Tensor(y_one_hot)
    return softmax_loss(h,y_).numpy(), np.mean(h.numpy().argmax(axis=1) != y)
