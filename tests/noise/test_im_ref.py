import numpy as np
import pytest

import pynam.dielectric
import pynam.noise.im_ref
from pynam.baskets import CalculationParams


@pytest.fixture
def im_ref_p_lindhard():
	params = CalculationParams(
		omega=1e9, v_f=2e6, omega_p=3.544907701811032e15, tau=1e-14
	)
	eps_l = pynam.dielectric.get_lindhard_dielectric(params)
	return pynam.noise.im_ref.get_im_ref_p(eps_l)


@pytest.mark.parametrize(
	"test_input,expected",
	[
		# u   im_ref_p_l(u)
		# needs to be close in range around 1/z, so from 1e4 to 1e8
		# (1e4, 1.821722334939806e-8), 1e4 is too far off still
		(1e5, 1.602855764970752e-8),
		(1e6, 1.704326041013161e-8),
		(1e7, 2.674124312031195e-8),
		(1e8, 7.441319151047531e-8),
	],
)
def test_im_ref_p_lindhard(im_ref_p_lindhard, test_input, expected):
	actual = im_ref_p_lindhard(test_input)

	np.testing.assert_allclose(
		actual,
		expected,
		rtol=1e-4,
		err_msg="imrp is inaccurate for Lindhard case",
		verbose=True,
	)
