import numpy as np
import theano
import theano.tensor as T

x_var = T.fmatrix('x')
w_var = T.fmatrix('w')
b_var = T.fvector('b')

z = T.dot(x_var, w_var) + b_var  # broadcasting!
y = theano.tensor.nnet.sigmoid(z)  # theano.tensor.tanh(z)
f = theano.function(inputs=[x_var, w_var, b_var], outputs=[z, y])

x = np.arange(5, dtype=np.float32).reshape((1, 5))
w = np.arange(5*10, dtype=np.float32).reshape((5, 10))
b = np.ones(10, dtype=np.float32)


n_in = 5
n_out = 10

W = theano.shared(np.asarray(
     np.random.normal(loc=0.0, scale=np.sqrt(1.0/n_in), size=(n_in, n_out)),
   dtype=theano.config.floatX),
        name='w', borrow=True)

b = theano.shared(np.zeros((n_out, ), dtype=theano.config.floatX), name='b', borrow=True)

print f(x, w, b)
