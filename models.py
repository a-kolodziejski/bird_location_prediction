"""
    Implementations of three seq-to-seq models: RNN, LSTM, and GRU.
"""
    
    # Imports
import torch
    
    
    
class LSTMModel(torch.nn.Module):
    '''
    Implements LSTM model with arbitrary number of LSTM layers.

    Args:
        input_dim (int): dimensionality (number of features) of input
        hidden_dim (int): dimensionality of hidden vector
        output_dim (int): dimensionality of output vector
        num_layers (int): number of LSTM layers   
        horizon (int): number of timesteps into the future to predict
    '''
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, horizon):
        super().__init__()
        self.horizon = horizon
        self.lstm = torch.nn.LSTM(input_size = input_dim, 
                                    hidden_size = hidden_dim,
                                    num_layers = num_layers,
                                    batch_first = True)
        self.linear = torch.nn.Linear(in_features = hidden_dim, out_features = output_dim)

    def forward(self, X):
        '''
        Implements forward propagation of input X. 
        X is expected to be of the shape: (batch_size, window_size, number_of_features)
        '''
        # Pass X through lstm
        output, _ = self.lstm(X)
        # Pick only last horizon many vectors from output
        output_vector = output[:, -self.horizon:, :]
        # Pass it through linear layer
        out = self.linear(output_vector)
        
        return out


class GRUModel(torch.nn.Module):
    '''
    Implements GRU model with arbitrary number of GRU layers.

    Args:
        input_dim (int): dimensionality (number of features) of input
        hidden_dim (int): dimensionality of hidden vector
        output_dim (int): dimensionality of output vector
        num_layers (int): number of GRU layers   
        horizon (int): number of timesteps into the future to predict
    '''
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, horizon):
        super().__init__()
        self.horizon = horizon
        self.gru = torch.nn.GRU(input_size = input_dim, 
                                    hidden_size = hidden_dim,
                                    num_layers = num_layers,
                                    batch_first = True)
        self.linear = torch.nn.Linear(in_features = hidden_dim, out_features = output_dim)

    def forward(self, X):
        '''
        Implements forward propagation of input X. 
        X is expected to be of the shape: (batch_size, window_size, number_of_features)
        '''
        # Pass X through lstm
        output, _ = self.gru(X)
        # Pick only last horizon many vectors from output
        output_vector = output[:, -self.horizon:, :]
        # Pass it through linear layer
        out = self.linear(output_vector)
        
        return out
        

class RNNModel(torch.nn.Module):
    '''
    Implements RNN model with arbitrary number of layers.

    Args:
        input_dim (int): dimensionality (number of features) of input
        hidden_dim (int): dimensionality of hidden vector
        output_dim (int): dimensionality of output vector
        num_layers (int): number of RNN layers   
        horizon (int): number of timesteps into the future to predict
    '''
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, horizon):
        super().__init__()
        self.horizon = horizon
        self.rnn = torch.nn.RNN(input_size = input_dim, 
                                    hidden_size = hidden_dim,
                                    num_layers = num_layers,
                                    batch_first = True)
        self.linear = torch.nn.Linear(in_features = hidden_dim, out_features = output_dim)

    def forward(self, X):
        '''
        Implements forward propagation of input X. 
        X is expected to be of the shape: (batch_size, window_size, number_of_features)
        '''
        # Pass X through lstm
        output, _ = self.rnn(X)
        # Pick only last horizon many vectors from output
        output_vector = output[:, -self.horizon:, :]
        # Pass it through linear layer
        out = self.linear(output_vector)
        
        return out
        