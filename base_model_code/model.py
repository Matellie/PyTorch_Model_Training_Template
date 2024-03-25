import torch.nn as nn
from torch import sigmoid

class Model(nn.Module):

    def __init__(self):
        """
        Model parameters:
        - layers
        """
        super(Model, self).__init__()

    def forward(self, x):
        """
        Model forward:
        - use the layers
        - return model output
        """
        pass

class LinearRegression(nn.Module):
    """
    Linear regression is used to predict the value of a variable based on the value of one or more other
    variables.
    """
    def __init__(self, input_dim, output_dim):
        super(LinearRegression, self).__init__()

        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)
    
    def get_name(self):
        return 'LinearRegression'

class LogisticRegression(nn.Module):
    """
    Logistic regression estimates the probability of an event occurring (P=1) or not occurring (P=0),
    based on a given dataset of independent variables. Since the outcome is a probability, the dependent
    variable is bounded between 0 and 1.
    """
    def __init__(self, n_imput_features):
        super(LogisticRegression, self).__init__()

        self.linear = nn.Linear(n_imput_features, 1)
        self.sigmoid = sigmoid()

    def forward(self, x):
        x = self.linear(x)
        x = self.sigmoid(x)
        return x
    
    def get_name(self):
        return 'LogisticRegression'

class SimpleNeuralNet(nn.Module):
    """
    Simple neural net with one hidden layer using the ReLU activation function.
    """
    def __init__(self, input_size, hidden_size, num_classes):
        super(SimpleNeuralNet, self).__init__()

        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x

    def get_name(self):
        return 'SimpleNeuralNet'