#############################################################################################################
# Code is a reimplementation of the code provided by the authors of the paper:                              #
# Neural Joint Entropy Estimation                                                                           #
# https://github.com/YuvalShalev/NJEE                                                                       #
#############################################################################################################


import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class ModelBasicClassification(nn.Module):
    def __init__(self, input_shape, n_classes, hidden_size=25):
        super(ModelBasicClassification, self).__init__()
        self.l1 = nn.Linear(input_shape, hidden_size)
        #self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, n_classes)
        self.output = nn.Softmax(dim=1) # normalise output to be a probability distribution over classes

    def forward(self, x):
        x = torch.relu(self.l1(x))  # linear + ReLU
        #x = torch.relu(self.l2(x))  # linear + ReLU
        x = self.l3(x)              # linear
        output = self.output(x)     # softmax
        return output

def discretise(input: np.ndarray, n_bins):
    """
    Discretize data to convert into one-hot probability vectors
    """
    x = np.arange(0, len(input)) # x coordinates are simply the indeces of the input
    y = input.reshape(-1) # y coordinates are the input values reshaped to a 1D array
    discretised, _, __ = np.histogram2d(x, y, bins=[len(input), n_bins])
    return discretised


def NJEE_estimate_condentropy(
        X:np.ndarray, 
        Y:list[np.ndarray], 
        device, 
        n_bins=250,
        hidden_size=25,
        epochs=1001,
        batch_size=1000,
        ):
    """
    Estimate conditional entropy H(X|Y) using NJEE
    Y can in fact be a list of multiple variables
    """
    assert isinstance(device, torch.device)
    assert isinstance(X, np.ndarray)
    assert isinstance(Y, list)
    assert all(isinstance(y, np.ndarray) for y in Y)
    assert all(X.shape[0] == y.shape[0] for y in Y)
        
    xdims = X.shape[1]
    ydims = sum([y.shape[1] for y in Y])

    # Init list of models and optimizers for each dimension
    model_lst_cond = []
    for m in range(0, xdims):
        model_lst_cond.append(ModelBasicClassification(input_shape=m + ydims, n_classes=n_bins, hidden_size=hidden_size))

    for m in range(0, xdims): # send models to device
        model_lst_cond[m].to(device)

    opt_lst_cond = []
    for m in range(0, xdims):
        opt_lst_cond.append(optim.Adam(model_lst_cond[m].parameters(), lr=0.001))


    # Prepare input data
    input_lst = [[] for _ in range(xdims)]
    for n in range(xdims):
    # input is the 1, 2, ..., n-1-th dimension of X, and all of Y
        base_input = np.concatenate(Y, axis=1)
        if   n == 0:
            # corresponds to the input "Y" for H(X_1|Y)
            input_full = base_input
        elif n >= 1:
            # corresponds to the input "X_1, ..., X_{n-1}, Y" for H(X_n|X_1, ..., X_{n-1}, Y)
            input_full = np.concatenate([X[:, :n], base_input], axis=1)
        
        for i in range(0, X.shape[0], batch_size):
            input = input_full[i:i + batch_size]

            input = torch.tensor(input, dtype=torch.float32).contiguous().to(device)
            input_lst[n].append(input)


    # Prepare target data
    target_lst = [[] for _ in range(xdims)]
    for n in range(xdims):
        # target is a discretized version of the n-th dimension of X
        target_full = X[:, n].reshape(-1, 1)
        target_full = discretise(target_full, n_bins)
        
        for i in range(0, X.shape[0], batch_size):
            target = target_full[i:i + batch_size]

            target = torch.tensor(target, dtype=torch.long).contiguous().to(device)
            target_lst[n].append(target)

    # Estimate conditional entropy H(X|Y) using the decomposition formula:
    """
    H(X|Y) = H(X_1|Y) + H(X_2|X_1, Y) + H(X_3|X_1, X_2, Y) + ... + H(X_n|X_1, ..., X_{n-1}, Y)
    """
    # Entropy will be summed across all dimensions
    H_perdim = np.zeros(xdims)
    
    for n in range(xdims):
        # Track losses along each epoch
        losses = []
        # Train model
        for i in range(epochs):
            # Select a random batch as input
            index = np.random.randint(0, len(input_lst[n]))
            input = input_lst[n][index]
            target = target_lst[n][index]
            
            # Get output
            opt_lst_cond[n].zero_grad()
            output = model_lst_cond[n](input)
            
            # Compute the cross-entropy between the output and the target
            loss = -torch.mean(torch.sum(target * torch.log(output + 1e-12), dim=1))
            losses.append(loss.item())
            
            # Backpropagate and update weights
            loss.backward()
            opt_lst_cond[n].step()

            
            if i % 500 == 0:
                print(f'Dimension {n+1}/{xdims}, Epoch {i}, Loss: {np.round(np.mean(losses[-25:]),4)}', end='\r')

        H_perdim[n] = np.mean(losses[-25:])

    # sum across all dimensions
    H_xy = np.sum(H_perdim)
        
    return np.round(H_xy,4)


def TE_njee(var_from, var_to, device, n_bins, epochs=1001, hidden_size=25, batch_size=1000):
    """
    Estimate the transfer entropy from var_from to var_to using NJEE
    """
    X_past = var_from[:-1]
    Y_past = var_to[:-1]
    Y_future = var_to[1:]

    H_YGY = NJEE_estimate_condentropy(
        Y_future, [Y_past], 
        device, n_bins, epochs=epochs, hidden_size=hidden_size, batch_size=batch_size
        )
    print(f'H(Y|Y-) = {H_YGY}                                          ', end='\r')
    
    H_YGXY = NJEE_estimate_condentropy(
        Y_future, [X_past, Y_past], 
        device, n_bins, epochs=epochs, hidden_size=hidden_size, batch_size=batch_size
        )
    print(f'H(Y|Y-) = {H_YGY}, H(Y|X-,Y-) = {H_YGXY}                   ', end='\r')
    
    TE_X2Y = np.round(H_YGY - H_YGXY, 4)
    print(f'H(Y|Y-) = {H_YGY}, H(Y|X-,Y-) = {H_YGXY}, TE(X->Y) = {TE_X2Y}'        )

    return TE_X2Y

def TE_njee_batch(var_from_lst, var_to_lst, device, n_bins, epochs=1001, hidden_size=25, batch_size=1000):
    """
    Estimate the transfer entropy from var_from to var_to using NJEE
    Concatenates data across batches in a way that does not corrupt the result
    """
    X_past = []
    Y_past = []
    Y_future = []
    for var_from, var_to in zip(var_from_lst, var_to_lst):
        X_past.append(var_from[:-1])
        Y_past.append(var_to[:-1])
        Y_future.append(var_to[1:])

    X_past = np.concatenate(X_past, axis=0)
    Y_past = np.concatenate(Y_past, axis=0)
    Y_future = np.concatenate(Y_future, axis=0)


    H_YGY = NJEE_estimate_condentropy(
        Y_future, [Y_past], 
        device, n_bins, epochs=epochs, hidden_size=hidden_size, batch_size=batch_size
        )
    print(f'H(Y|Y-) = {H_YGY}                                          ', end='\r')
    
    H_YGXY = NJEE_estimate_condentropy(
        Y_future, [X_past, Y_past], 
        device, n_bins, epochs=epochs, hidden_size=hidden_size, batch_size=batch_size
        )
    print(f'H(Y|Y-) = {H_YGY}, H(Y|X-,Y-) = {H_YGXY}                   ', end='\r')
    
    TE_X2Y = np.round(H_YGY - H_YGXY, 4)
    print(f'H(Y|Y-) = {H_YGY}, H(Y|X-,Y-) = {H_YGXY}, TE(X->Y) = {TE_X2Y}'        )

    return TE_X2Y