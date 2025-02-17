# Activation Function 


import torch

class ReLU:
    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        Applies the ReLU activation function.
        ReLU(x) = max(0, x)
        """
        return torch.maximum(torch.zeros_like(x), x)

    def backward(self, delta: torch.tensor, x: torch.tensor) -> torch.tensor:
        """
        Computes the gradient of ReLU.
        ReLU'(x) = 1 if x > 0 else 0
        """
        return delta * (x > 0).float()


class LeakyReLU:
    def __init__(self, alpha=0.1):
        """
        Initializes the LeakyReLU activation function with a specified alpha value.
        """
        self.alpha = alpha

    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        Applies the Leaky ReLU activation function.
        LeakyReLU(x) = x if x > 0 else alpha * x
        """
        return torch.where(x >= 0, x, self.alpha * x)

    def backward(self, delta: torch.tensor, x: torch.tensor) -> torch.tensor:
        """
        Computes the gradient of Leaky ReLU.
        LeakyReLU'(x) = 1 if x > 0 else alpha
        """
        return delta * torch.where(x >= 0, torch.ones_like(x), self.alpha * torch.ones_like(x))



# initialize

def initialize(self) -> None:
     """
    Initialize all biases to zero and weights using Xavier initialization.
    """
    for i in range(self.num_layers):
        d_in = self.layer_sizes[i]
        d_out = self.layer_sizes[i + 1]
        w_range = np.sqrt(6 / (d_in + d_out))
        W = torch.empty(d_in, d_out, device=device).uniform_(-w_range, w_range)
        self.weights.append(W)
        b = torch.zeros(1, d_out, device=device) 
        self.biases.append(b)
    
    
# forward


def forward(self, x: torch.tensor) -> torch.tensor:
        """
        Forward propagation through all layers.
        Applies activation function to all layers except the last one.
        """
        self.features = [x.to(device)]  

        for i in range(self.num_layers):  
            x = torch.matmul(x, self.weights[i]) + self.biases[i]
            x = self.activation_function.forward(x)  
            self.features.append(x) 
        return x
    

# backward 

def backward(self, delta: torch.Tensor) -> None:
        '''
        This function should backpropagate the provided delta through the entire MLP, and update the weights according to the hyper-parameters
        stored in the class variables.
        '''
        
        for i in reversed(range(self.num_layers)):
            x = self.features[i]

            delta = self.activation_function.backward(delta,self.features[i+1])
            
            dW = torch.matmul(x.T,delta) / self.batch_size
            db = torch.sum(delta, dim=0, keepdim=True) / self.batch_size

            self.weights[i] -= self.learning_rate * dW
            self.biases[i] -= self.learning_rate * db
            delta = torch.matmul(delta,self.weights[i].T)


# Extra credit 

# Activation - Tanh and Sigmoid

import torch

class Tanh:
    def forward(self, x: torch.tensor) -> torch.tensor:
        return  (torch.exp(x) - torch.exp(-x)) / (torch.exp(x) + torch.exp(-x))
    
    def backward(self, delta: torch.tensor, x: torch.tensor) -> torch.tensor:
        tanh_org = (torch.exp(x) - torch.exp(-x)) / (torch.exp(x) + torch.exp(-x))
        tanh_derivative = 1 - tanh_org * tanh_org
        return delta * tanh_derivative

class Sigmoid:
    def forward(self, x: torch.tensor) -> torch.tensor:
        return 1 / (1 + torch.exp(-x))
    
    def backward(self, delta: torch.tensor, x: torch.tensor) -> torch.tensor:
        sig_x_org = 1 / (1 + torch.exp(-x))
        sig_x_derivative = (sig_x_org * (1 - sig_x_org))
        return delta * sig_x_derivative