"""
membership functions used in the fuzzy system
"""
import torch

def gauss(x, m, sigma):
    """
    gaussian membership function
    exp(-(x-m)^2/(2*sigma^2))
    :param x: independent variable
    :param m: center
    :param sigma: spread
    :return: membership values
    """
    return (-(x - m) ** 2 / (2 * sigma ** 2)).exp()

def generalized_cgmf(x, m, sigma, lambdas):
    """
    Generalized Compose Gaussian Membership Function
    :param x: input
    :param m: center
    :param sigma: spread
    :param lambdas: the newly added parameter in this membership function
    :return:
    """
    gate_value = 1 - torch.exp(- 10 * lambdas ** 2)
    height = 10 * gate_value + 0.001

    return (- height * (1 - (-(x - m) ** 2 / (2 * height * sigma ** 2)).exp())).exp()
