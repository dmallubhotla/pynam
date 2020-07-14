import pynam.dielectric
import numpy as np
import pytest
from pynam.baskets import CalculationParams


def get_common_lindhard_dielectric():
	params = CalculationParams(omega=1e9, omega_p=3.5e15, tau=1e-14, v_f=2e6)
	return pynam.dielectric.get_lindhard_dielectric(params)


@pytest.mark.parametrize("test_input,expected", [
	(10, -1222.185185062794 + 1.2249999998777178e8j),
	(1000, 16924.14814718176 + 1.2250000020552777e8j),
	(1e8, 83.687499999706 + 0.00022417398943752126j)
])
def test_lindhard_dielectric(test_input, expected):

	eps_to_test = get_common_lindhard_dielectric()

	np.testing.assert_almost_equal(
		eps_to_test(test_input), expected,
		decimal=6, err_msg='b function is off'
	)


@pytest.mark.parametrize("test_input,expected", [
	((100, 100), -883.3001542404703 + 1.2566370613549341e8j),
	((100, 1e5), 5.827225842825694e7 + 3.933446612656656e7j),
	((100, 1e10), 1.0084823001646925 + 2.0013975538629039e-10j),
	((100, 1e7), 8483.300121667038 + 0.6340397839154446)
])
def test_zeta_pi_lindhard_dielectric(zeta_p_i_epsilon, test_input, expected):
	u, y = test_input
	actual = zeta_p_i_epsilon(np.sqrt(u**2 + y**2))

	np.testing.assert_allclose(
		actual, expected,
		rtol=10**3.8, err_msg='lindhard dielectric differs from Mathematica'
	)


@pytest.fixture
def zeta_p_i_epsilon():
	params = CalculationParams(omega=1e9, omega_p=3.544907701811032e15, tau=1e-14, v_f=2e6)
	return pynam.dielectric.get_lindhard_dielectric(params)
