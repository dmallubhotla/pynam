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
