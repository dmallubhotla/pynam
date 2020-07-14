import numpy as np
import pytest

import pynam.dielectric
import pynam.noise.zeta
from pynam.baskets import CalculationParams


@pytest.fixture
def zeta_p_integrand_lindhard():
	params = CalculationParams(omega=1e9, v_f=2e6, omega_p=3.544907701811032e15, tau=1e-14)
	eps_l = pynam.dielectric.get_lindhard_dielectric(params)
	return pynam.noise.zeta.get_zeta_p_integrand(eps_l)


@pytest.mark.parametrize("test_input,expected", [
	# y    u     zeta_p_i(u, y)
	((100, 100), -6.891930153028566e-13 - 7.957747045025948e-9j),
	((1e5, 100), -1.0057257267146669e-10 - 4.0591966623027983e-13j),
	((100, 1e5), 1.1789175285399862e-8 - 7.957833322596519e-9j)
])
def test_zeta_p_integrand_lindhard(zeta_p_integrand_lindhard, test_input, expected):
	actual = zeta_p_integrand_lindhard(*test_input)

	np.testing.assert_allclose(
		actual, expected,
		rtol=1e-7, err_msg='Zeta_p is inaccurate for Lindhard case', verbose=True
	)


@pytest.fixture
def zeta_p_lindhard():
	params = CalculationParams(omega=1e9, v_f=2e6, omega_p=3.544907701811032e15, tau=1e-14)
	eps_l = pynam.dielectric.get_lindhard_dielectric(params)
	return pynam.noise.zeta.get_zeta_p_function(eps_l)


@pytest.mark.parametrize("test_input,expected", [
	# u    zeta_p(u)
	(1, 0.000199609 - 0.000199608j),
])
def test_zeta_p(zeta_p_lindhard, test_input, expected):
	actual = zeta_p_lindhard(test_input)

	np.testing.assert_allclose(
		actual, expected,
		rtol=1e-7, err_msg='Zeta_p is inaccurate for Lindhard case', verbose=True
	)
