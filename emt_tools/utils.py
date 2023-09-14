"""
Helper functions for emt_tools TODO: Probably have to refactor this file later
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch
from matplotlib.collections import PatchCollection
import networkx as nx
import matplotlib.animation


def get_grounded_bases(W_uh, W_hh, W_hy, s, alpha=1, h_simulated=None, f_operator=None, strength=1):
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

    W_uh = W_hy.T @ np.linalg.inv(W_hy @ W_hy.T)

    W_hy_star = np.linalg.pinv(W_hy)
    original_s = s

    # # reduce dimensionality of W_hh by removing directions with <1 magnitude eigenvalues
    # W_hh_original = W_hh.copy()
    # eig_vals, eig_vecs = np.linalg.eig(W_hh)
    # eigvals_matrix = np.diag(eig_vals)
    # eigvals_matrix[np.absolute(eigvals_matrix) < 1] = 0
    # W_hh = eig_vecs @ eigvals_matrix @ np.linalg.inv(eig_vecs)

    Psi = [W_uh]  # note that Psi is reversed list of variable memories

    for k in range(s - 1, 0, -1):
        Psi.append(np.linalg.matrix_power(W_hh, strength*s - k) @ W_uh)

    Psi.reverse()
    Psi = np.concatenate(Psi, axis=1).squeeze()
    # print(Psi.shape)

    # indices_to_add = np.array(list(range(task_dimension * s)))
    # optimize the basis (seems like NNs learn a compressed basis)
    if f_operator is not None:
        if torch.is_tensor(f_operator):
            f_operator = f_operator.cpu().detach().numpy()

        indices_array = np.array(list(range(task_dimension))).reshape((-1, 1)) + 1
        indices_array = (np.abs(f_operator) @ indices_array).flatten()

        indices_to_add = []

        # re-initialize s to the new number of variables
        raw_indices_array = indices_array.copy()
        for i in range(s):
            raw_indices_array[i * task_dimension:(i + 1) * task_dimension] = (((s - i) * task_dimension -
                                                                               raw_indices_array[i * task_dimension:(
                                                                                                                            i + 1) * task_dimension]) *
                                                                              (raw_indices_array[i * task_dimension:(
                                                                                                                            i + 1) * task_dimension] > 0))  # remove indices not utilized
        # print(raw_indices_array, np.max(raw_indices_array))
        s = np.ceil(np.max(raw_indices_array) / task_dimension).astype(np.int_)

        # print("new s: {}".format(s))

        for variable_index in range(s):
            variable_indices = indices_array[variable_index * task_dimension:(variable_index + 1) * task_dimension]
            variable_indices = variable_indices[variable_indices > 0]

            # print(variable_index, variable_indices - 1)

            indices_to_add.append(variable_index * task_dimension + variable_indices.astype(np.int_))
            if variable_index > 0:
                indices_to_add[-1] = np.append(indices_to_add[-1], task_dimension + indices_to_add[-2])
                indices_to_add[-1] = np.unique(indices_to_add[-1])

        indices_to_add = np.concatenate(indices_to_add).flatten()
        indices_to_add -= 1

        # print(indices_to_add)
        Psi = Psi[:, indices_to_add]
    else:
        indices_to_add = np.array(list(range(task_dimension * s)))

    # normalize the basis - TODO: think why no normalization
    Psi = Psi / np.linalg.norm(Psi, axis=0)

    # compute the dual basis
    Psi_star = np.linalg.inv(Psi.T @ Psi) @ Psi.T
    # Psi_star = Psi_star / np.linalg.norm(Psi_star, axis=0)
    # Psi_star = Psi.T @ np.linalg.inv(Psi @ Psi.T)

    # expanded basis
    Psi_expanded = np.zeros((hidden_dim, original_s * task_dimension))
    Psi_star_expanded = np.zeros((original_s * task_dimension, hidden_dim))

    Psi_expanded[:, indices_to_add] = Psi
    Psi_star_expanded[indices_to_add, :] = Psi_star

    return Psi_expanded, Psi_star_expanded


def plot_evolution_in_bases(u_history, h_history, y_history, Psi_star, Phi,
                            task_dimension, s, filename="animation.gif"):
    """
    Plot the evolution of the RNN states in the basis for which Psi_star is the dual
    """
    neuron_radius = 0.1

    vb_circle_radius = 2 * neuron_radius * task_dimension

    hstate_x = []
    hstate_y = []
    inputstate_x = [-vb_circle_radius - 2.5 * neuron_radius - 1.5] * task_dimension
    inputstate_y = [-vb_circle_radius - 1 - i * (2 * neuron_radius + neuron_radius // 4) for i in range(task_dimension)]

    outputstate_x = [vb_circle_radius + 2.5 * neuron_radius + 1.5] * task_dimension
    outputstate_y = [-vb_circle_radius - 1 - i * (2 * neuron_radius + neuron_radius // 4) for i in range(task_dimension)]

    radii = [neuron_radius] * (s * task_dimension)

    # print(h_history.shape, Psi_star.shape)

    h_basis = h_history @ Psi_star.T

    delta_theta = 2 * np.pi / s
    cur_theta = (s - 1) * delta_theta - np.pi / 2
    for i in range(s):
        x_cur = vb_circle_radius * np.cos(cur_theta)
        y_cur = vb_circle_radius * np.sin(cur_theta) + (task_dimension // 2) * neuron_radius

        for j in range(task_dimension):
            hstate_x.append(x_cur)
            hstate_y.append(y_cur)

            y_cur -= (2 * neuron_radius + neuron_radius // 4)

        cur_theta -= delta_theta

    # Build plot
    fig, ax = plt.subplots(figsize=(6, 4))
    hstate_patches = []
    inputstate_patches = []
    outputstate_patches = []

    for x1, y1, r in zip(hstate_x, hstate_y, radii):
        circle = Circle((x1, y1), r)
        hstate_patches.append(circle)

    for x1, y1, r in zip(inputstate_x, inputstate_y, radii[:task_dimension]):
        circle = Circle((x1, y1), r)
        inputstate_patches.append(circle)

    for x1, y1, r in zip(outputstate_x, outputstate_y, radii[:task_dimension]):
        circle = Circle((x1, y1), r)
        outputstate_patches.append(circle)

    ## add hiddenstate rectangle
    ax.add_patch(FancyBboxPatch((-vb_circle_radius - 0.5, -vb_circle_radius - task_dimension * neuron_radius - 0.5),
                           2*(vb_circle_radius + 0.5), 2*vb_circle_radius + task_dimension * neuron_radius + 1.5,
                           color="black", alpha=0.1))

    # add these circles to a collection
    p_hstate = PatchCollection(hstate_patches, cmap="coolwarm", alpha=1.0)
    ax.add_collection(p_hstate)

    p_inputstate = PatchCollection(inputstate_patches, cmap="coolwarm", alpha=1.0)
    ax.add_collection(p_inputstate)

    p_outputstate = PatchCollection(outputstate_patches, cmap="coolwarm", alpha=1.0)
    ax.add_collection(p_outputstate)

    ax.set_xlim(-vb_circle_radius - 0.5, vb_circle_radius + 0.5)
    ax.set_ylim(-vb_circle_radius - (task_dimension // 2) * neuron_radius - 3.0,
                vb_circle_radius + (task_dimension // 2) * neuron_radius + 1.5)

    ax.axis("equal")
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    ax.set_title("Step {} ({} phase)".format(1, "input"), fontweight="bold")

    plt.tight_layout()

    def update(num):
        p_hstate.set_array(Psi_star @ h_history[num, :].flatten())  # set new color colors
        p_inputstate.set_array(u_history[num].flatten())
        p_outputstate.set_array(y_history[num].flatten())

        # Scale plot ax
        phase = "input"
        if num >= s:
            phase = "output"
        ax.set_title("Step {} ({} phase)".format(num + 1, phase), fontweight="bold")

        p_hstate.set_clim(-0.1, 0.1)
        p_inputstate.set_clim(-0.3, 0.3)
        p_outputstate.set_clim(-0.3, 0.3)

        return p_hstate, p_inputstate, p_outputstate

    ani = matplotlib.animation.FuncAnimation(fig, update,
                                             frames=h_history.shape[0],
                                             interval=1000, repeat=True)
    ani.save(filename, writer='imagemagick', fps=1)

    pass


def construct_phi_from_f_operator(f_operator):
    """
    Construct the phi from the f_operator
    :param f_operator: np.ndarray of size (d, sd)
    :return: np.ndarray of size (sd, sd)
    """
    d, sd = f_operator.shape
    ## construct phi
    phi = np.eye(sd)
    phi = np.roll(phi, d)

    phi[:, :d] = 0
    phi[-d:, :] = f_operator

    return phi


def spectral_comparison(operator1, operator2, threshold=1-1e-3):
    """
    Compare the spectral properties of two linear operators.
    The eigenvalues less than threshold is full removed and the spectrum is compared.

    R = \{ (\lamda^1_i, \lambda^2_i) \forall i \in max(|\lambda^1|, |\lambda^2|) \}
    argmin_{R} \sum_{|Arg(\lambda^1_i) - Arg(\lambda^2_i)|}

    :param operator1: np.ndarray
    :param operator2: np.ndarray
    :param threshold: float (default 1-1e-3)
    :return:

    """

    eigenvalues1, eigenvectors1 = np.linalg.eig(operator1)
    eigenvalues2, eigenvectors2 = np.linalg.eig(operator2)

    # remove eigenvalues less than threshold
    evals1_reduced = eigenvalues1[np.absolute(eigenvalues1) > threshold]
    evals2_reduced = eigenvalues2[np.absolute(eigenvalues2) > threshold]

    # minimum rank operator is operator1
    if evals1_reduced.shape[0] != evals2_reduced.shape[0]:
        return -1

    n = evals1_reduced.shape[0]

    # compute the spectral distance
    evals1_vec = np.zeros((n, 2))
    evals2_vec = np.zeros((n, 2))

    evals1_vec[:, 0] = evals1_reduced.real
    evals1_vec[:, 1] = evals1_reduced.imag
    evals1_vec /= np.linalg.norm(evals1_vec, axis=1).reshape((-1, 1))

    evals2_vec[:, 0] = evals2_reduced.real
    evals2_vec[:, 1] = evals2_reduced.imag
    evals2_vec /= np.linalg.norm(evals2_vec, axis=1).reshape((-1, 1))

    err = 0
    # compute the complex argument error
    for i in range(n):
        delta = np.arccos((evals1_vec[i:i+1].reshape((1, -1)) @ evals2_vec.T).flatten())
        delta_argmin = np.argmin(np.absolute(delta))

        err += np.absolute(delta[delta_argmin])

        # remove the used eigenvalues
        evals2_vec = np.delete(evals2_vec, delta_argmin, axis=0)

    return err/n
