import torch
import numpy as np
# print(torch.__version__)  # Check PyTorch version --> 2.8.0
# print(torch.cuda.is_available())  # returns True if CUDA is available, but i have M1 thus i need to use MPS instead of CUDA
# print(torch.backends.mps.is_available())  # Check if MPS is available for Apple Silicon --> True
# print(torch.backends.mps.is_built())  # Check if MPS is built in this PyTorch installation --> True

# Exercise 0.1 - making tensors
x = torch.rand(5, 3) # creates a tensor with 5 rows, 3 columns, and random values between 0 and 1 as the entries.
x = torch.tensor([[5,1,2], [1,1,1], [0,0,0]]) # just like a numpy array?
x = torch.zeros(5, 3) # just like numpy?
x = torch.ones(5, 3) # all just ones?

data = [[1,0], [0,1]]
x_data = torch.tensor(data)

x = torch.ones_like(x_data) # in the same shape as the "data" tensor, but with all ones
x = torch.zeros_like(x_data, dtype=torch.float) # in the same shape as the "data" tensor, but with all zeroes, makes sure it reads it as floats rather than int?
x = torch.rand_like(x_data, dtype=torch.float) # in the same shape as the "data" tensor, but with all random
# print(x)


# Exercise 0.2 - converting between numpy and torch
np_data = np.array(data)
np_into_torch = torch.from_numpy(np_data) # turns the numpy array into a torch tensor
back_again = np.array(np_into_torch) # converts back into a numpy array again
or_back_again = np_into_torch.numpy() # converts back into a numpy array again
# print(np_data)
# print(np_into_torch)

# Exercise 0.3 - attributes
tensor = torch.rand(3, 4)
# print(f"Shape of tensor: {tensor.shape}") # Shape of tensor: torch.Size([3, 4])
# print(f"Datatype of tensor: {tensor.dtype}") # Datatype of tensor: torch.float32
# print(f"Device tensor is stored on: {tensor.device}") # Device tensor is stored on: cpu

# Exercise 0.4 - simple interactions
tensor = torch.ones(4, 4)
tensor[:,1] = 0 #same slicing as numpy, where its rows then columns. This makes the 1 column all 0s
t1 = torch.cat([tensor, tensor, tensor], dim=1) #concatenate
# Multiplying element-wise or "mul"
tensor * tensor
tensor.mul(tensor)
#Matrix multiplication or "matmul"
tensor @ tensor.T
tensor.matmul(tensor.T)

# Exercise 1: Autograd
"""Forward Propagation: In forward prop, the NN makes its best guess about the correct output. 
It runs the input data through each of its functions to make this guess.

Backward Propagation: In backprop, the NN adjusts its parameters proportionate to the error in its guess. 
It does this by traversing backwards from the output, collecting the derivatives of the error with respect 
to the parameters of the functions (gradients), and optimizing the parameters using gradient descent. For a 
more detailed walkthrough of backprop, check out this video from 3Blue1Brown.

Let's take a look at a single training step. For this example, we load a pretrained resnet18 model from torchvision. 
We create a random data tensor to represent a single image with 3 channels, and height & width of 64, and its 
corresponding label initialized to some random values. Label in pretrained models has shape (1,1000).
"""

from torchvision.models import resnet18, ResNet18_Weights
model = resnet18(weights=ResNet18_Weights.DEFAULT)
data = torch.rand(1, 3, 64, 64)
labels = torch.rand(1, 1000)

prediction = model(data) # forward pass
loss = (prediction - labels).sum()
loss.backward() # backward pass
optim = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9) #out optimiser, stochastic dragient descent, learning rate and momentum
optim.step() #gradient descent intiating

# Jacobian explanation at https://docs.pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html

# Q1.1 - Create a tensor x with requires_grad=True and compute y = x^2. What is dy/dx?

x = torch.tensor([[1, 1, 1], [1, 2, 3]], dtype=torch.float32, requires_grad=True) # make them float32
y = x**2

# method 1: flatten to a scalar
y.sum().backward() # using .sum() to combine all the components, which backward requires to backprop and store to .grad

# method 2: define a function, and use the jacobian functional
def f(x):
    return x**2

J = torch.autograd.functional.jacobian(f, x) # in this case, a rank-4 tensor J.shape=(2,3,2,3) with 2,3 for y and 2,3 for x... you need to do some linear algebra to map back to a 2x3 matrix


# Exercise 2 - Linear Regression
"""
Neuural networks can be constructed using the torch.nn package. nn depends on autograd to define models. and differentiate them.
A typical training procedure for a neural network is as follows:
- Define the neural network that has some learnable parameters (or weights)
- Iterate over a dataset of inputs
- Process input through the network
- Compute the loss (how far is the output from being correct)
- Propagate gradients back into the network’s parameters
- Update the weights of the network, typically using a simple update rule: weight = weight - learning_rate * gradient
"""
# 2.1 define network

import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self): # where we define the layers of the nn
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5) #from these 6 feature maps, makes 16 with another 5x5 conv
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 5*5 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10) #outputs 100 classes

    def forward(self, input):   #input shape: (N, 1, 32, 32) (batch of 32×32 grayscale images, N = batch size).
        # Convolution layer C1: 1 input image channel, 6 output channels,
        # 5x5 square convolution, it uses RELU activation function, and
        # outputs a Tensor with size (N, 6, 28, 28), where N is the size of the batch
        c1 = F.relu(self.conv1(input))       # applies the rectified linear unit element wise (doing the convolution?)
        # Subsampling layer S2: 2x2 grid, purely functional,
        # this layer does not have any parameter, and outputs a (N, 6, 14, 14) Tensor
        s2 = F.max_pool2d(c1, (2, 2))
        # Convolution layer C3: 6 input channels, 16 output channels,
        # 5x5 square convolution, it uses RELU activation function, and
        # outputs a (N, 16, 10, 10) Tensor
        c3 = F.relu(self.conv2(s2))
        # Subsampling layer S4: 2x2 grid, purely functional,
        # this layer does not have any parameter, and outputs a (N, 16, 5, 5) Tensor
        s4 = F.max_pool2d(c3, 2)
        # Flatten operation: purely functional, outputs a (N, 400) Tensor
        s4 = torch.flatten(s4, 1)
        # Fully connected layer F5: (N, 400) Tensor input,
        # and outputs a (N, 120) Tensor, it uses RELU activation function
        f5 = F.relu(self.fc1(s4))
        # Fully connected layer F6: (N, 120) Tensor input,
        # and outputs a (N, 84) Tensor, it uses RELU activation function
        f6 = F.relu(self.fc2(f5))
        # Gaussian layer OUTPUT: (N, 84) Tensor input, and
        # outputs a (N, 10) Tensor
        output = self.fc3(f6)
        return output


net = Net()

# Net(
#   (conv1): Conv2d(1, 6, kernel_size=(5, 5), stride=(1, 1))
#   (conv2): Conv2d(6, 16, kernel_size=(5, 5), stride=(1, 1))
#   (fc1): Linear(in_features=400, out_features=120, bias=True)
#   (fc2): Linear(in_features=120, out_features=84, bias=True)
#   (fc3): Linear(in_features=84, out_features=10, bias=True)
# )


# autograd will define the backprop for us

params = list(net.parameters())
len(params) # 10
params[0].size()  # conv1's .weight: torch.Size([6, 1, 5, 5])

# 2.2 - input a gray scale image 32x32
input = torch.randn(1, 1, 32, 32)
out = net(input)
# tensor([[ 0.0873,  0.1252,  0.0004,  0.0720, -0.0709, -0.0070,  0.0762,  0.0591,
#          -0.0041, -0.0450]], grad_fn=<AddmmBackward0>)

net.zero_grad() #reset the grad so that the last one isn't being added to our weights and biases
out.backward(torch.randn(1, 10))

"""Recap:
torch.Tensor - A multi-dimensional array with support for autograd operations like backward(). Also holds the gradient w.r.t. the tensor.
nn.Module - Neural network module. Convenient way of encapsulating parameters, with helpers for moving them to GPU, exporting, loading, etc.
nn.Parameter - A kind of Tensor, that is automatically registered as a parameter when assigned as an attribute to a Module.
autograd.Function - Implements forward and backward definitions of an autograd operation. Every Tensor operation creates at least a single Function node that connects to functions that created a Tensor and encodes its history.
"""

# 2.3 - loss function. Obvisously there are many different loss functions, e.g. MSELoss is a mean squared error loss function.

output = net(input)
target = torch.randn(10)  # a dummy target, for example
target = target.view(1, -1)  # make it the same shape as output
criterion = nn.MSELoss()

loss = criterion(output, target) # tensor(1.3175, grad_fn=<MseLossBackward0>)

# once again, zero the autograd
net.zero_grad()     # zeroes the gradient buffers of all parameters

# print('conv1.bias.grad before backward')
# print(net.conv1.bias.grad)
# None
loss.backward()

# print('conv1.bias.grad after backward')
# print(net.conv1.bias.grad)
# tensor([ 0.0104,  0.0018,  0.0097, -0.0102, -0.0091,  0.0047])

# 2.4 - update weights. Stochastic gradient descent simplist version follows: weight = weight - learning_rate * gradient

import torch.optim as optim

# create your optimizer
optimiser = optim.SGD(net.parameters(), lr=0.01)

# in your training loop:
optimiser.zero_grad()   # zero the gradient buffers
output = net(input)
loss = criterion(output, target)
loss.backward()
optimiser.step()    # Does the update