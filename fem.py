import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse
import scipy.sparse.linalg
from scipy.interpolate import lagrange
import pandas as pd
from basis_functions import Linear, Quadratic


GAUSS_DEGREE = 4


def exact_solution_a(x, Q=1):
    return 1 - np.exp(Q*x)


def conductivity_a(x, Q=1):
    return np.exp(-Q*x)


def exact_solution_b(x, Q=1, L=1):
    if isinstance(x, (list,np.ndarray)):
        res = np.zeros(len(x))
        res[x <= -L/3] = -Q*(1000*x[x <= -L/3] + 333*L)
        res[x >= -L/3] = -Q * x[x >= -L/3]
        return res
    else:
        if x <= -L/3:
            return -Q*(1000*x + 333*L)
        if x >= -L/3:
            return -Q*x


def conductivity_b(x, Q=1, L=1):
    if isinstance(x, (list, np.ndarray)):
        res = np.empty(len(x))
        res[x <= -L/3] = 0.001
        res[x >= -L/3] = 1
        return res
    else:
        if x <= -L/3:
            return 0.001
        if x >= -L/3:
            return 1


def FEM(N, conductivity=conductivity_a, exact_solution=exact_solution_a, L=1, Q=1, quadratic=False):
    """
    :param N: number of elements
    :param conductivity: function to get hydraulic conductivity
    :param exact_solution: exact solution function
    :param L: lower boundary
    :param Q: Neumann boundary condition
    :param quadratic: bool, if True use quadratic basis functions else linear
    :return: nodes, mesh step, solution in nodes
    """
    h = (0-(-L)) / N  # mesh step
    N_nodes = N+1  # number of nodes
    nodes = np.linspace(-L, 0, N_nodes, endpoint=True)
    # Basic tests
    assert N_nodes == len(nodes)
    assert np.isclose(nodes[1] - nodes[0], h)

    basis_func = Linear
    if quadratic:
        basis_func = Quadratic
    basis_func_der = basis_func.get_functions_der()

    # Generate quadrature on reference element
    ref_ele = [0, 1]
    pt, w = np.polynomial.legendre.leggauss(GAUSS_DEGREE)
    quad_pt = (pt + 1) / 2 * (ref_ele[1] - ref_ele[0]) + ref_ele[0]
    quad_w = w * (ref_ele[1] - ref_ele[0]) / 2

    if quadratic:
        N_nodes = N_nodes + N_nodes-1

    A = scipy.sparse.csr_matrix((N_nodes, N_nodes))
    b = np.zeros(N_nodes-1)

    for k, node in enumerate(nodes[:-1]):
        if quadratic:
            A_i = k*2
        else:
            A_i = k
        M = np.zeros((len(basis_func_der), len(basis_func_der), len(quad_pt)))
        for i in range(len(basis_func_der)):
            for j in range(i, len(basis_func_der)):
                x_c = basis_func.points_from_ref_to_ele(node, h, quad_pt)
                M[i, j, :] += quad_w * conductivity(x_c, Q) * basis_func_der[i](quad_pt) * basis_func_der[j](quad_pt)
                M[j, i, :] = M[i, j, :]
        M *= 1/h
        A[A_i:(A_i+len(basis_func_der)), A_i:(A_i+len(basis_func_der))] += np.sum(M, axis=-1)

    b[0] = Q
    A = A[:-1, :-1]  # Basis functions in last node should be zero
    p_h = scipy.sparse.linalg.spsolve(A, b)  # Solve sparse linear system
    p_h = list(p_h)
    p_h.append(0)

    #plot_res(basis_func, nodes, h, p_h, exact_solution, quadratic, L)
    return nodes, h, p_h


def plot_res(basis_func, nodes, h, p_h, exact_solution, quadratic=False, L=1):
    x = np.linspace(-L, 0, 250)
    basis_func_der = basis_func.get_functions()
    resolution_per_ele = 1
    ele_x_values = []
    ele_y_values = []

    for k, node in enumerate(nodes[:-1]):
        if quadratic:
            k = k * 2
        x_ref = np.linspace(0, 1, resolution_per_ele)
        ele_x = basis_func.points_from_ref_to_ele(node, h, x_ref)
        ele_x_values.extend(ele_x)

        s = np.zeros(resolution_per_ele)
        for phi in basis_func_der:
            s += p_h[k] * phi(x_ref)
        ele_y_values.extend(s)

    plt.plot(ele_x_values, ele_y_values, label="aprox")
    plt.plot(x, exact_solution(x), ":", label="exact")
    plt.legend()
    #plt.yscale('log')
    plt.show()

# def plot_exact_solution():
#     x = np.linspace(-1, 0, 1000)
#     #print("exact solution ", exact_solution_a(x))
#     plt.plot(x, exact_solution_a(x), label="exact solution a")
#     # exact_solution_b_vec = np.vectorize(exact_solution_b)
#     # plt.plot(x, exact_solution_b_vec(x), label="exact solution b")
#     plt.show()


def compute_err_L2(nodes, p_h, config):
    """
    Compute L2 norm
    :param nodes: mesh nodes
    :param p_h: approx solution vector of p
    :param config: dict, contains exact solution function, type of basis functions, ...
    :return: L2 norm
    """
    p_exact = config["exact_solution"]

    pt, w = np.polynomial.legendre.leggauss(GAUSS_DEGREE)
    err = 0
    for i in range(len(nodes)-1):
        # Rescale quadrature from [-1, 1] to the particular element
        quad_pt = (pt + 1) / 2 * (nodes[i+1] - nodes[i]) + nodes[i]
        quad_w = w * (nodes[i+1] - nodes[i]) / 2

        if config["quadratic"]:
            poly = lagrange([nodes[i], (nodes[i] + nodes[i+1])/2, nodes[i+1]], [p_h[i*2], p_h[2*i+1], p_h[2*i+2]])
        else:
            poly = lagrange([nodes[i], nodes[i + 1]], [p_h[i], p_h[i + 1]])

        et = np.dot((p_exact(quad_pt) - poly(quad_pt))**2, quad_w)
        err += et

    return np.sqrt(err)


def test_convergence(config):
    N_list = [10, 20, 40, 80, 160, 320]

    L2_errors = []
    steps_h = []
    conv_rates = []
    for i, N in enumerate(N_list):
        nodes, h, p_h = FEM(N, **config)
        L2_errors.append(compute_err_L2(nodes, p_h, config))
        steps_h.append(h)
        if i > 0:
            conv_rate = np.log(L2_errors[i-1] / L2_errors[i]) / np.log(steps_h[i-1] / steps_h[i])
            conv_rates.append(conv_rate)

    print("N ele ", N_list)
    print("steps_h ", steps_h)
    print("conv rates ", conv_rates)

    #plt.plot(N_list, L2_errors)
    plt.plot(steps_h, L2_errors)
    if config["quadratic"]:
        plt.plot(steps_h, np.array(steps_h)**3, ":", label=r"$h^3$")
    else:
        plt.plot(steps_h, np.array(steps_h) ** 2, ":", label=r"$h^2$")
    plt.ylabel(r"$\Vert p - \pi_h p \Vert_{2}$")
    plt.xlabel("h")
    plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    plt.savefig("{}_{}_conv_rate.pdf".format(config["quadratic"], config["exact_solution"]))
    plt.show()


if __name__ == '__main__':
    config = {"conductivity": conductivity_a, "exact_solution": exact_solution_a, "L": 1, "Q": 1, "quadratic": False}
    config = {"conductivity": conductivity_a, "exact_solution": exact_solution_a, "L": 1, "Q": 1, "quadratic": True}
    # config = {"conductivity": conductivity_b, "exact_solution": exact_solution_b, "L": 1, "Q": 1, "quadratic": False}
    # config = {"conductivity": conductivity_b, "exact_solution": exact_solution_b, "L": 1, "Q": 1, "quadratic": True}
    test_convergence(config)

    #N = 4  # number of elements
    #FEM(N, **config)

