"""
Helper functions for emt_tools
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.animation


def get_grounded_bases(W_uh, W_hh, W_hy, s, alpha=1, h_simulated=None, f_operator=None):
    """
    Runs the variable memory computation algorithm grounding the hidden state with input

    :param W_uh: np.ndarray
    :param W_hh: np.ndarray
    :param W_hy: np.ndarray (default: 1)
    :param s: int
        number of steps in the "input phase"
    :param alpha: float in [0, 1]
        alpha controls the amount of grounding between input and outputs.
        alpha = 0: fully grounded in output
        alpha = 1 (default): fully grounded in input
    :param h_simulated: np.ndarray (default: None)
        simulated history of hidden states for computing orthogonal basis. If None, the basis is not computed

    :return:
        Psi: variable memory bases (expanded to full dimensionality)
        Psi_star: dual of the variable memory bases (expanded to full dimensionality)
    """

    # get dimensions

    hidden_dim, task_dimension = W_uh.shape

    W_hy_star = np.linalg.pinv(W_hy)
    original_s = s

    Psi = [alpha * W_uh + (1 - alpha) * W_hy_star]  # note that Psi is reversed list of variable memories

    for k in range(s - 1, 0, -1):
        print(k)
        Psi.append(alpha * np.linalg.matrix_power(W_hh, s - k) @ W_uh +
                   (1 - alpha) * np.linalg.matrix_power(W_hh.T, k) @ W_hy_star)

    Psi.reverse()
    Psi = np.concatenate(Psi, axis=1).squeeze()
    print(Psi.shape)

    # optimize the basis (seems like NNs learn a compressed basis)
    if f_operator is not None:
        if torch.is_tensor(f_operator):
            f_operator = f_operator.cpu().detach().numpy()

        indices_array = np.array(list(range(task_dimension))).reshape((-1, 1))+1
        indices_array = (np.abs(f_operator) @ indices_array).flatten()

        indices_to_add = []

        # re-initialize s to the new number of variables
        raw_indices_array = indices_array.copy()
        for i in range(s):
            raw_indices_array[i*task_dimension:(i+1)*task_dimension] = (((s-i)*task_dimension -
                                                                        raw_indices_array[i*task_dimension:(i+1)*task_dimension]) *
                                                                        (raw_indices_array[i*task_dimension:(i+1)*task_dimension] > 0))  # remove indices not utilized
        print(raw_indices_array, np.max(raw_indices_array))
        s = np.ceil(np.max(raw_indices_array)/task_dimension).astype(np.int_)

        print("new s: {}".format(s))

        for variable_index in range(s):
            variable_indices = indices_array[variable_index*task_dimension:(variable_index+1)*task_dimension]
            variable_indices = variable_indices[variable_indices > 0]

            print(variable_index, variable_indices-1)

            indices_to_add.append(variable_index * task_dimension + variable_indices.astype(np.int_))
            if variable_index > 0:
                indices_to_add[-1] = np.append(indices_to_add[-1], task_dimension + indices_to_add[-2])
                indices_to_add[-1] = np.unique(indices_to_add[-1])

        indices_to_add = np.concatenate(indices_to_add).flatten()
        indices_to_add -= 1

        print(indices_to_add)
        Psi = Psi[:, indices_to_add]
    else:
        indices_to_add = np.array(list(range(task_dimension*s)))

    # normalize the basis
    Psi = Psi / np.linalg.norm(Psi, axis=0)

    # compute the dual basis and normalize
    Psi_star = np.linalg.pinv(Psi)
    Psi_star = Psi_star / np.linalg.norm(Psi_star, axis=0)

    # expanded basis
    Psi_expanded = np.zeros((hidden_dim, original_s*task_dimension))
    Psi_star_expanded = np.zeros((original_s * task_dimension, hidden_dim))

    Psi_expanded[:, indices_to_add] = Psi
    Psi_star_expanded[indices_to_add, :] = Psi_star

    return Psi_expanded, Psi_star_expanded


def plot_evolution_in_bases(u_history, h_history, y_history, Psi_star, task_dimension, s):
    """
    Plot the evolution of the RNN states in the basis for which Psi_star is the dual
    """
    n_variables = s
    dimension_variable = task_dimension

    variable_basis = Psi_star[:n_variables*dimension_variable, :]


    pass
