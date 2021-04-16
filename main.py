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


def conductivity_1(x, Q=1):
    return np.exp(-Q*x)


class Linear():

    @staticmethod
    def points_from_ref_to_ele(x1, h, x):
        """
        :param x1: element starting point
        :param h: length of element
        :param x: reference element point(s)
        :return: element points
        """
        return x1 + h*x

    @staticmethod
    def points_from_ele_to_ref(x1, h, x):
        """
        :param x1: element starting point
        :param h: length of element
        :param x: reference element point(s)
        :return: element points
        """
        return (x-x1)/h

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


def FEM(N):
    L = 1
    Q = 1
    h = (0-(-L)) / (N-1)
    print("h ", h)
    x_ele = np.linspace(-L, 0, N, endpoint=True)
    basis_func = Linear

    print("x_ele ", x_ele)
    ref_ele = [0, 1]
    pt, w = np.polynomial.legendre.leggauss(GAUSS_DEGREE)

    pt = (pt + 1) / 2 * (ref_ele[1] - ref_ele[0]) + ref_ele[0]
    w = w * (ref_ele[1] - ref_ele[0]) / 2

    #pt = 0.5 * (pt + 1) * (ref_ele[1] - ref_ele[0]) + ref_ele[0]  # rescale quandrature points to ref element [0, 1]

    print("pt ", pt)
    print("w ", w)

    A = np.zeros((N, N))
    b = np.zeros(N-1)

    for k, x_k in enumerate(x_ele[:-1]):
        print("x_k ", x_k)
        # the particular element is range [x_k, x_{k+1}]
        ele_ref_x = Linear.points_from_ref_to_ele(x_k, h, pt)
        print("ele ref x ", ele_ref_x)

        phi_1, phi_2, phi_1_der, phi_2_der = Linear.phi_ref_1, Linear.phi_ref_2, Linear.phi_ref_1_der, Linear.phi_ref_2_der

        M = np.zeros((2, 2))

                # def scipy_quad_1(x):
                #     x_c = Linear.points_from_ref_to_ele(x_k, h, x)
                #     return conductivity_1(x_c, Q) * phi_1_der(x, h) * h
                #
                # def scipy_quad(x):
                #     x_c = Linear.points_from_ref_to_ele(x_k, h, x)
                #     print("x: {}, x_c:{}".format(x, x_c))
                #     return phi_1_der(x, h) * phi_2_der(x, h) * h * conductivity_1(x_c, Q)
                #
                # if k == len(x_ele) - 2:
                #     M[i, j] = integrate.quad(scipy_quad_1, ref_ele[0], ref_ele[1])[0]
                # else:
                #     res = integrate.quad(scipy_quad, ref_ele[0], ref_ele[1])
                #     print("res ", res)
                #     M[i, j] = res[0]

        for quad_pt, quad_w in zip(pt, w):
            x_c = Linear.points_from_ref_to_ele(x_k, h, quad_pt)
            print("x_k:{}, x_c:{}, quad_pt: {}".format(x_k, x_c, quad_pt))
            # for i in range(2):
            #     for j in range(2):
            if k == len(x_ele) - 2:
                M[0, 0] += quad_w * conductivity_1(x_c, Q) * phi_1_der(quad_pt, h) * phi_1_der(quad_pt, h) * h
                M[1, 0] += 0
                M[0, 1] += 0
                M[1, 1] += 0
            else:
                M[0, 0] += quad_w * conductivity_1(x_c, Q) * phi_1_der(quad_pt, h) * phi_1_der(quad_pt, h) * h
                M[0, 1] += quad_w * conductivity_1(x_c, Q) * phi_1_der(quad_pt, h) * phi_2_der(quad_pt, h) * h
                M[1, 0] += quad_w * conductivity_1(x_c, Q) * phi_2_der(quad_pt, h) * phi_1_der(quad_pt, h) * h
                M[1, 1] += quad_w * conductivity_1(x_c, Q) * phi_2_der(quad_pt, h) * phi_2_der(quad_pt, h) * h

        #M = M * 0.5 * (ref_ele[1] - ref_ele[0])
        print("M ", M)
        #M = M/h
        A[k:(k+2), k:(k+2)] += M[:,:]
        #b[k:(k+2)] += h

    b[0] += Q

    A = A[:-1, :-1]

    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print("A ")
        print(pd.DataFrame(A))

    print("A[k:k+1, k:k+1] ", A[k:k + 1, k:k + 1])
    print("b ", b)
    print("A ", A)


    p_coefs = np.linalg.solve(A, b)
    print("p ", p_coefs)


    x = np.linspace(-L, 0, 10)
    x = x_ele
    res = []
    #print("x ", x)
    for x_i in x:
        s = 0
        for k in range(N-1):
            #print("k: {}, x: {}".format(k, x_i))
            #x_ref = Linear.points_from_ele_to_ref(x_ele[k], h, x_i)
            s += p_coefs[k] * linear_basis_functions(k, x_i, x_ele, h)
        res.append(s)

    print("res ", res)

    plt.plot(x, res)
    plt.show()

    # for e in (elements):
    #     # prevest na referencni element
    #     for quad:
    #         for i:
    #             for j:
    #                 # pridani prvku do lokalni matice M
    #
    #      A += M
    #


def linear_basis_functions(i, x, x_h, h):

    if i == 0:
        if x < x_h[i]:
            return 0
        elif x_h[i] <= x < x_h[i + 1]:
            return 1 - (x - x_h[i]) / h
        elif x >= x_h[i + 1]:
            return 0
    else:
        if x < x_h[i-1]:
            return 0
        elif x_h[i-1] <= x < x_h[i]:
            return (x - x_h[i-1])/h
        elif x_h[i] <= x < x_h[i+1]:
            return 1 - (x-x_h[i])/h
        elif x >= x_h[i+1]:
            return 0

    print("i ", i)
    print("x_h ", x_h)
    print("x: {}, x_h[i]: {}, x_h[i-1]: {}".format(x, x_h[i], x_h[i - 1]))


def exact_solution_b(x, Q=1, L=1):
    #@TODO: which value should be set in -L/3?
    if x <= -L/3:
        return -Q*(1000*x + 333*L)
    return -Q*x


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
    N = 50
    FEM(N)

    plot_exact_solution()
    #plot_basis_functions(N)
    # plot_reference_basis_fce()
