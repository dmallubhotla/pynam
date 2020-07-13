import numpy as np
import pynam.util.complex_quad


def test_complex_quad():
	actual = pynam.util.complex_quad.complex_quadrature(lambda x: x ** 2 + 1j * x ** 3, 0, 6)[0]
	# int_1^6 dx x^2 + i x^3 should equal (1/3)6^3 + (i/4)6^4
	np.testing.assert_almost_equal(
		actual, (6**3)/3 + 1j*(6**4)/4,
		decimal=7, err_msg='complex quadrature is broken', verbose=True
	)
