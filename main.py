import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt
import pandas as pd

GAUSS_DEGREE = 10#@TODO: How to properly determine gauss degree


# def reference_basis_fce(x, x_s, x_r, d=1):
#     value = 1
#     for i in range(d):
#         value *= (x-x_s)/ (x_s - x_r)
#
#     return value

def exact_solution_a(x, Q=1):
    return 1 - np.exp(Q*x)


def conductivity_a(x, Q=1):
    return np.exp(-Q*x)


def exact_solution_b(x, Q=1, L=1):
    if isinstance(x, (list,np.ndarray)):
        res = np.zeros(len(x))
        print("x shape ", x.shape)
        print("res shape ", res.shape)
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

# def conductivity_1_der(x, Q=1):
#     return -Q*np.exp(-Q*x)


class BasisFunctions:
    @staticmethod
    def points_from_ref_to_ele(x1, h, x):
        """
        :param x1: element starting point
        :param h: length of element
        :param x: reference element point(s)
        :return: element points
        """
        return x1 + h * x

    @staticmethod
    def points_from_ele_to_ref(x1, h, x):
        """
        :param x1: element starting point
        :param h: length of element
        :param x: reference element point(s)
        :return: element points
        """
        return (x - x1) / h


class Linear(BasisFunctions):
    @staticmethod
    def phi_ref_1(x):
        return 1-x

    @staticmethod
    def phi_ref_2(x):
        return x

    @staticmethod
    def phi_ref_1_der(x=None, h=1):
        #print("phi 1 der h ", h)
        return (-1/h)

    @staticmethod
    def phi_ref_2_der(x=None, h=1):
        return (1/h)

    @staticmethod
    def get_functions_der():
        return [Linear.phi_ref_1_der, Linear.phi_ref_2_der]

    @staticmethod
    def get_functions():
        return [Linear.phi_ref_1, Linear.phi_ref_2]


class Quadratic(BasisFunctions):
    @staticmethod
    def phi_ref_1(x):
        return 2*(x**2) - 3*x +1

    @staticmethod
    def phi_ref_2(x):
        return -4*(x**2) + 4*x

    @staticmethod
    def phi_ref_3(x):
        return 2 * (x ** 2) - x

    @staticmethod
    def phi_ref_1_der(x=None, h=1):
        return (4*x-3)*(1/h)

    @staticmethod
    def phi_ref_2_der(x=None, h=1):
        return (-8 * x + 4) * (1/h)

    @staticmethod
    def phi_ref_3_der(x=None, h=1):
        return (4*x-1)*(1/h)

    @staticmethod
    def get_functions_der():
        return [Quadratic.phi_ref_1_der, Quadratic.phi_ref_2_der, Quadratic.phi_ref_3_der]

    @staticmethod
    def get_functions():
        return [Quadratic.phi_ref_1, Quadratic.phi_ref_2, Quadratic.phi_ref_3]

def FEM(N, quadratic=False):
    #quadratic = True
    exact_solution = exact_solution_b
    conductivity = conductivity_b
    L = 1
    Q = 1
    N_nodes = N+1
    h = (0-(-L)) / (N_nodes)
    print("h ", h)
    x_ele = np.linspace(-L, 0, N_nodes, endpoint=True)
    basis_func = Linear
    if quadratic:
        basis_func = Quadratic

    print("x_ele ", x_ele)
    ref_ele = [0, 1]
    pt, w = np.polynomial.legendre.leggauss(GAUSS_DEGREE)

    pt = (pt + 1) / 2 * (ref_ele[1] - ref_ele[0]) + ref_ele[0]
    w = w * (ref_ele[1] - ref_ele[0]) / 2

    print("pt ", pt)
    print("w ", w)

    print("N ", N)
    #exit()

    if quadratic:
        N_nodes = N_nodes + N_nodes-1
    print("N ", N)
    A = np.zeros((N_nodes, N_nodes))
    b = np.zeros(N_nodes-1)

    functions = basis_func.get_functions_der()

    print("x ele ", x_ele)
    z =0
    for k, x_k in enumerate(x_ele[:-1]):
        #k = k
        if quadratic:
            z = 2*k
        else:
            z = k
        print("K index ", k)
        #print("x_k ", x_k)
        # the particular element is range [x_k, x_{k+1}]
        #ele_ref_x = Linear.points_from_ref_to_ele(x_k, h, pt)
        #print("ele ref x ", ele_ref_x)

        M = np.zeros((len(functions), len(functions)))
        for i in range(len(functions)):
            for j in range(i, len(functions)):
                for quad_pt, quad_w in zip(pt, w):
                    x_c = basis_func.points_from_ref_to_ele(x_k, h, quad_pt)
                    #print("x_k:{}, x_c:{}, quad_pt: {}".format(x_k, x_c, quad_pt))

                    if k == len(x_ele) - 2:
                        #print("LAST k: {}".format(k))
                        #if i == 1 or j == 1:
                        if i == len(functions)-1 or j == len(functions)-1:
                            print("i: {}, j: {}".format(i, j))
                            M[i,j] += 0
                        else:
                            M[i, j] += quad_w * conductivity(x_c, Q) * functions[i](quad_pt, h) * functions[j](quad_pt, h) * h
                    else:
                        M[i, j] += quad_w * conductivity(x_c, Q) * functions[i](quad_pt, h) * functions[j](quad_pt, h) * h
                        M[j, i] = M[i,j]

        k_index = z
        A[k_index:(k_index+len(functions)), k_index:(k_index+len(functions))] += M

    b[0] = Q
    if quadratic:
        A = A[:-1, :-1]
        b = b#[:-1]
    else:
        A = A[:-1, :-1]

    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print("A ")
        print(pd.DataFrame(A))

    # print("A[k:k+1, k:k+1] ", A[k:k + 1, k:k + 1])
    # print("b ", b)
    # print("A ", A)

    p_coefs = np.linalg.solve(A, b)
    print("p ", p_coefs)


    x = np.linspace(-L, 0, 250)
    functions = basis_func.get_functions()
    resolution_per_ele = 10
    ele_x_values = []
    ele_y_values = []

    for k, x_k in enumerate(x_ele[:-1]):
        if quadratic:
            k = k*2
        #print("x_k ", x_k)
        x_ref = np.linspace(0, 1, resolution_per_ele)
        ele_x = basis_func.points_from_ref_to_ele(x_k, h, x_ref)
        #print("element_x ", ele_x)

        ele_x_values.extend(ele_x)

        s = np.zeros(resolution_per_ele)
        for phi in functions:
            s += p_coefs[k] * phi(x_ref)
        ele_y_values.extend(s)

    #plt.plot(x_ele, p_coefs, label="aprox")
    plt.plot(ele_x_values, ele_y_values, label="aprox")
    plt.plot(x, exact_solution(x), label="exact")
    plt.legend()
    plt.yscale('log')
    plt.show()


# def linear_basis_functions(i, x, x_h, h):
#
#     if i == 0:
#         if x < x_h[i]:
#             return 0
#         elif x_h[i] <= x < x_h[i + 1]:
#             return 1 - (x - x_h[i]) / h
#         elif x >= x_h[i + 1]:
#             return 0
#     else:
#         if x < x_h[i-1]:
#             return 0
#         elif x_h[i-1] <= x < x_h[i]:
#             return (x - x_h[i-1])/h
#         elif x_h[i] <= x < x_h[i+1]:
#             return 1 - (x-x_h[i])/h
#         elif x >= x_h[i+1]:
#             return 0
#
#     print("i ", i)
#     print("x_h ", x_h)
#     print("x: {}, x_h[i]: {}, x_h[i-1]: {}".format(x, x_h[i], x_h[i - 1]))


# def exact_solution_b(x, Q=1, L=1):
#     #@TODO: which value should be set in -L/3?
#     if x <= -L/3:
#         return -Q*(1000*x + 333*L)
#     return -Q*x


def plot_exact_solution():
    x = np.linspace(-1, 0, 1000)
    #print("exact solution ", exact_solution_a(x))
    plt.plot(x, exact_solution_a(x), label="exact solution a")
    # exact_solution_b_vec = np.vectorize(exact_solution_b)
    # plt.plot(x, exact_solution_b_vec(x), label="exact solution b")
    plt.show()


def plot_fce(i, x, x_h, h):
    res = []
    for x_k in x:
        res.append(linear_basis_functions(i, x_k, x_h, h))
    #print("res ", res)
    plt.plot(x, res)


def plot_basis_functions(N):
    h = 1/(N-1)
    x_h = np.linspace(-1, 0, N, endpoint=True)
    #x_h = [0.2, 0.4, 0.6, 0.8]

    # x_h = [-2].extend(x_h)
    # x_h.append(2)

    print("x_h ", x_h)
    x = np.linspace(-1, 0, 1000)
    #x = [-1, -0.8, -0.75]

    plot_fce(0, x, x_h, h)
    plot_fce(1, x, x_h, h)
    plot_fce(2, x, x_h, h)
    plot_fce(3, x, x_h, h)
    #plot_fce(4, x, x_h, h)
    plt.show()

def local_matrix(N):
    A = np.empty((N, N))
    for k in range(N):
        pass
        #A[k:k+1, k:k+1] = M


def plot_quadratic_basis_fce():
    x = np.linspace(0, 1, 200)

    plt.plot(x, Quadratic.phi_ref_1(x), label="ref 1")
    plt.plot(x, Quadratic.phi_ref_2(x), label="ref 2")
    plt.plot(x, Quadratic.phi_ref_3(x), label="ref 3")
    plt.legend()

    plt.show()


# def plot_reference_basis_fce():
#     N = 4
#     L = 1
#     h = L / (N + 1)
#     x_h = np.linspace(-1, 0, N)
#     # x_h = [0.2, 0.4, 0.6, 0.8]
#     x = np.linspace(-1, 1, 200)
#
#     res = []
#     for x_k in x:
#         #res.append(reference_basis_fce(x_k, x_h[0], x_h[1], d=1))
#         res.append(reference_basis_fce(x_k, x_h[0], x_h[1], d=1))
#     print("res ", res)
#     plt.plot(x, res)
#     plt.show()



if __name__ == '__main__':
    N = 500# number of elements
    # plot_quadratic_basis_fce()
    # exit()

    FEM(N, quadratic=True)

    #plot_exact_solution()
    #plot_basis_functions(N)
    # plot_reference_basis_fce()
