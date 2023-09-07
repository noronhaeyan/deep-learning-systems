"""Operator implementations."""
import pdb
from numbers import Number
from typing import Optional, List

import numpy as np

from .autograd import NDArray
from .autograd import Op, Tensor, Value, TensorOp
from .autograd import TensorTuple, TensorTupleOp
import numpy

# NOTE: we will import numpy as the array_api
# as the backend for our computations, this line will change in later homeworks
import numpy as array_api


class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION

        return a ** self.scalar

        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION

        if self.scalar == 1:
            return out_grad

        return mul_scalar( power_scalar(out_grad, self.scalar - 1), self.scalar)

        ### END YOUR SOLUTION


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION

        return a/b

        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION

        a , b = node.inputs

        #temp = ndl.Tensor([4,6,1.5])
        #print(power_scalar(a, 1))
        #print(power_scalar(a, 2))
        return divide(out_grad, b), multiply(out_grad , divide(-a, power_scalar(b, 2)))

        ### END YOUR SOLUTION


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        ### BEGIN YOUR SOLUTION

        return a / self.scalar

        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION

        return out_grad / self.scalar

        ### END YOUR SOLUTION


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION

        if self.axes:
            axis1, axis2 = self.axes
        else:
            axis1, axis2 = len(a.shape)-1, len(a.shape)-2

        return array_api.swapaxes(a, axis1, axis2)


        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION

        return transpose(out_grad, axes = self.axes)


        ### END YOUR SOLUTION


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a.reshape(self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0]

        #pdb.set_trace()

        return out_grad.reshape(a.shape)
        ### END YOUR SOLUTION


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return array_api.broadcast_to(a, self.shape)

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION

        out_shape = self.shape
        in_shape = node.inputs[0].shape
        #print(in_shape)

        #pdb.set_trace()

        if in_shape in [tuple([1]), tuple()]:
            #print(summation(out_grad))
            return reshape(summation(out_grad), in_shape)

        sum_ax = []

        for i in range(len(out_shape)):
            if out_shape[i] != in_shape[i]:
                sum_ax.append(i)

        output = summation(out_grad, tuple(sum_ax))

        return reshape(output, in_shape)
        ### END YOUR SOLUTION


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.sum(a, axis = self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION

        out_shape = out_grad.shape
        in_shape = node.inputs[0].shape

        slow = 0
        fast = 0

        output = [1]*len(in_shape)

        while slow < len(out_shape):

            if in_shape[fast] == out_shape[slow]:
                output[fast] = in_shape[fast]
                slow += 1
                fast += 1
            else:
                fast += 1

        #pdb.set_trace()

        return broadcast_to(reshape(out_grad, tuple(output)), in_shape)
        ### END YOUR SOLUTION


def summation(a, axes=None):
    return Summation(axes)(a)

class Average(TensorOp):
    def compute(self, a):

        return array_api.mean(a)

    def gradient(self, out_grad, node):

        shape = node.inputs[0].shape

        N = shape[0]

        return broadcast_to(divide_scalar(out_grad, N), shape)

def average(a):
    return Average()(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return array_api.matmul(a, b)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        lhs, rhs = node.inputs

        output1 = matmul(out_grad, transpose(rhs))
        output2 = matmul(transpose(lhs), out_grad)

        #pdb.set_trace()

        if lhs.shape != output1.shape:
            n = len(output1.shape) - len(lhs.shape)

            axes = [i for i in range(n)]

            output1 = summation(output1, tuple(axes))

        if rhs.shape != output2.shape:
            n = len(output2.shape) - len(rhs.shape)

            axes = [i for i in range(n)]

            output2 = summation(output2, tuple(axes))

        return output1, output2
        ### END YOUR SOLUTION


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return -a
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return -out_grad
        ### END YOUR SOLUTION


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return np.log(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION

        shape = out_grad.shape

        ones = broadcast_to(Tensor([1.], dtype='float32'), shape)

        return multiply(divide(ones, node.inputs[0]), out_grad)
        ### END YOUR SOLUTION


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        print(array_api.exp(a))
        return array_api.exp(a)

        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION

        return multiply(out_grad, exp(node.inputs[0]))

        ### END YOUR SOLUTION


def exp(a):
    return Exp()(a)


# TODO
class ReLU(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION

        return a * (a > 0)

        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION

        input_array = node.inputs[0].realize_cached_data()

        indicator = Tensor(input_array > 0 , dtype= 'float32')

        return out_grad * indicator

        ### END YOUR SOLUTION


def relu(a):
    return ReLU()(a)

