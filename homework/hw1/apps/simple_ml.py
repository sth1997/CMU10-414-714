"""hw1/apps/simple_ml.py"""

import struct
import gzip
import numpy as np

import sys

sys.path.append("python/")
import needle as ndl


def parse_mnist(image_filename, label_filename):
    """Read an images and labels file in MNIST format.  See this page:
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

    # image
    g = gzip.GzipFile(mode = "rb", fileobj=open(image_filename, 'rb'))
    data = g.read()
    fmt = ">iiii"
    offset = 0
    magic_number, image_num, height, width = struct.unpack_from(fmt, data, offset)
    #print(magic_number, image_num, height, width) 
    offset += struct.calcsize(fmt)
    fmt = ">{}B".format(height * width)
    images = np.empty((image_num, height * width)).astype(np.float32)
    for img_num in range(image_num):
    #for img_num in range(1):
        pixel_list = struct.unpack_from(fmt, data, offset)
        offset += struct.calcsize(fmt)
        images[img_num] = np.array(pixel_list) / 255.0

    
    # label
    g = gzip.GzipFile(mode = "rb", fileobj=open(label_filename, 'rb'))
    data = g.read()
    fmt = ">ii"
    offset = 0
    magic_number, label_num = struct.unpack_from(fmt, data, offset)
    offset += struct.calcsize(fmt)
    fmt = ">B"
    labels = np.empty((label_num), dtype = np.uint8)
    for lb_num in range(label_num):
    #for lb_num in range(1):
        labels[lb_num] = struct.unpack_from(fmt, data, offset)[0]
        offset += struct.calcsize(fmt)

    return((images, labels))
    ### END YOUR SOLUTION


def softmax_loss(Z, y_one_hot):
    """Return softmax loss.  Note that for the purposes of this assignment,
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
    #return ((np.sum(np.log(np.sum(np.exp(Z), axis=1))) - np.sum(Z[np.arange(y.size), y]))/y.size)
    return ((ndl.log(ndl.exp(Z).sum(axes=(1,))).sum() - (y_one_hot * Z).sum()) / Z.shape[0])
    ### END YOUR SOLUTION


def nn_epoch(X, y, W1, W2, lr=0.1, batch=100):
    """Run a single epoch of SGD for a two-layer neural network defined by the
    weights W1 and W2 (with no bias terms):
        logits = ReLU(X * W1) * W1
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
    for i in range((X.shape[0] + batch - 1) // batch):
        X_batch = ndl.Tensor(X[i * batch : (i + 1) * batch, :])
        y_one_hot = np.zeros((batch, W2.shape[1]))
        y_one_hot[np.arange(batch), y[i * batch : (i + 1) * batch]] = 1
        y_batch = ndl.Tensor(y_one_hot)
        Z = ndl.relu(X_batch.matmul(W1)).matmul(W2)
        loss = softmax_loss(Z, y_batch)
        loss.backward()
        W1 = ndl.Tensor(W1.realize_cached_data() - lr * W1.grad.realize_cached_data())
        W2 = ndl.Tensor(W2.realize_cached_data() - lr * W2.grad.realize_cached_data())
    return W1, W2
    ### END YOUR SOLUTION


### CODE BELOW IS FOR ILLUSTRATION, YOU DO NOT NEED TO EDIT


def loss_err(h, y):
    """Helper function to compute both loss and error"""
    y_one_hot = np.zeros((y.shape[0], h.shape[-1]))
    y_one_hot[np.arange(y.size), y] = 1
    y_ = ndl.Tensor(y_one_hot)
    return softmax_loss(h, y_).numpy(), np.mean(h.numpy().argmax(axis=1) != y)
