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
	# (10, 0.00019960929309663014 - 0.00019927000998506335j),
	# (100, 0.0001996175250684056 - 0.0001654898843938523j),
	# (1e3, 0.0002003339895748246 + 0.003212370020888438j),
	# (1e4, 0.00028616168676982363 + 0.34096962141224463j),
	(1e5, 0.0025183067257958545 + 34.11087430547122j),
	(1e6, 0.026829658454640887 + 3411.0870128247902j),
	(1e7, 0.4292211181081069 + 341088.797211291j),
	(1e8, 14.348462224076096 + 3.391157983312813e7j)
])
def test_zeta_p(zeta_p_lindhard, test_input, expected):
	actual = zeta_p_lindhard(test_input)

	np.testing.assert_allclose(
		actual, expected,
		rtol=1e-4, err_msg='Zeta_p is inaccurate for Lindhard case', verbose=True
	)
