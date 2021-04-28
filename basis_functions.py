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
    def phi_ref_1_der(x=None):
        return -1

    @staticmethod
    def phi_ref_2_der(x=None):
        return 1

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
    def phi_ref_1_der(x):
        return 4*x-3

    @staticmethod
    def phi_ref_2_der(x):
        return -8 * x + 4

    @staticmethod
    def phi_ref_3_der(x):
        return 4*x-1

    @staticmethod
    def get_functions_der():
        return [Quadratic.phi_ref_1_der, Quadratic.phi_ref_2_der, Quadratic.phi_ref_3_der]

    @staticmethod
    def get_functions():
        return [Quadratic.phi_ref_1, Quadratic.phi_ref_2, Quadratic.phi_ref_3]

