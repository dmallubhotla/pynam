import pynam.util

from typing import Callable
import numpy as np


def get_zeta_p_integrand(eps: Callable[[float], complex]) -> Callable[[float, float], complex]:
	""" Gets the integrand function zeta_p_integrand(u, y).

	Returns zeta_p_integrand(u, y), a complex valued function of two momenta in units of vacuum wavelength.

	:param eps:
	:return:
	"""
	def zeta_p_integrand(u: float, y: float) -> complex:
		"""
		Here y and u are in units of vacuum wavelength, coming from Ford-Weber / from the EWJN noise expressions.
		:param u:
		:param y:
		:return:
		"""
		u2 = u ** 2
		y2 = y ** 2
		k2 = u2 + y2
		k = np.sqrt(k2)
		eps_value = eps(k)
		term_1 = y2 / (eps_value - k2)
		term_2 = u2 / eps_value
		return (term_1 + term_2) / k2

	return zeta_p_integrand


# def get_zeta_p_function(eps: Callable[[float], complex]):
# 	def zeta_p(u: float) -> complex:
# 		zeta_p_integrand = get_zeta_integrand(eps)
#
# 		integral_result = pynam.util.complex_quad(zeta_p_integrand, 0, np.inf)
#
# 		print(integral_result)
# 		integral = integral_result[0]
#
# 		return integral * 2j
#
# 	return zeta_p
