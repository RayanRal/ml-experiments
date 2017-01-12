import numpy as np

# here f(x) is our loss function
def eval_numeric_gradient(f, x):
    fx = f(x)  # evaluate function at original point

    grad = np.zeros(x.shape)  # starting gragient
    h = 0.00001  # the smallest value

    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:

        ix = it.multi_index
        old_value = x[ix]
        x[ix] = old_value + h # change old value by h
        fxh = f(x)  # evaluate function with x+h
        x[ix] = old_value # set the old value back

        grad[ix] = (fxh - fx) / h  # compute the gradient
        it.iternext()

    return grad


def L(a, b, c):
    return 0


def CIFAR10_loss_fun(W):
    return L(X_train, Y_train, W)

W = np.random.rand(10, 3073) * 0.001
df = eval_numeric_gradient(CIFAR10_loss_fun, W)

loss_original = CIFAR10_loss_fun(W) # the original loss

for step_size_log in [-10, -9, -8, -7, -6, -5,-4,-3,-2,-1]:
    step_size = 10 ** step_size_log
    W_new = W - step_size * df  # new weights, updated with help of gradient. Update in negative direction,
    # as we want loss function to decrease
    loss_new = CIFAR10_loss_fun(W_new) # the new loss, after weight update



# minibatch gradient descent
while True:
    data_batch = sample_training_data(data, batch_size = 256)  # taking 256 examples
    weights_grad = evaluate_gradient(loss_fun, data_batch, weights)
    weights += -step_size * weights_grad  # parameter update

