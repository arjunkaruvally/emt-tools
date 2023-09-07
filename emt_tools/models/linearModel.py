import numpy as np


class LinearModel:
    """
    EMT currently is limited to linear models.
    This class can be used to convert usual non-linear RNNs to an equivalent linear model.
    """
    def __init__(self, task_dimension, hidden_dimension):
        """
        Constructor for the LinearModel class.

        :param task_dimension:
            the dimension of the task space. That is, the dimensions u
            in the task u(t) = f(u(t-1), ... u(t-s))
        :param hidden_dimension:
            dimensions of the hidden state of the RNN
        """
        self.task_dimension = task_dimension
        self.hidden_dimension = hidden_dimension

        self.W_uh = np.zeros((self.task_dimension, self.hidden_dimension))
        self.W_hy = np.zeros((self.hidden_dimension, self.task_dimension))
        self.W_hh = np.zeros((self.hidden_dimension, self.hidden_dimension))

    def parse_simple_rnn(self, model):
        """
        Convert a simple Elman RNN (no bias), with a linear (without bias) output layer to the LinearModel class.

        :param model:
            pytorch rnn model with RNN module named 'rnn' and output layer named 'output'
        """
        self.W_uh = model.rnn.weight_ih_l0.detach().numpy()
        self.W_hh = model.rnn.weight_hh_l0.detach().numpy()
        self.W_hy = model.output.weight.detach().numpy()



