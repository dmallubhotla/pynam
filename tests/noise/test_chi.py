import numpy as np
import pytest

import pynam.dielectric
import pynam.noise.chi
from pynam.baskets import CalculationParams


@pytest.fixture
def chi_zz_e_lindhard():
	params = CalculationParams(omega=1e9, v_f=2e6, omega_p=3.544907701811032e15, tau=1e-14)
	eps_l = pynam.dielectric.get_lindhard_dielectric(params)
	return pynam.noise.chi.get_chi_zz_e(eps_l)


@pytest.mark.parametrize("test_input,expected", [
	# z   chi_zz_e_lindhard(z)
	(1e-5, 4.0249088868003124e6),
	(1e-6, 4.400474453780887e9),
])
def test_chi_zz_e_lindhard(chi_zz_e_lindhard, test_input, expected):
	actual = chi_zz_e_lindhard(test_input)

	np.testing.assert_allclose(
		actual, expected,
		rtol=1e-3, err_msg='chi_zz_e is inaccurate for Lindhard case', verbose=True
	)
