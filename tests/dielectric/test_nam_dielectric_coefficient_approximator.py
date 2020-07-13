import pytest
import numpy as np
import pynam.dielectric.nam_dielectric_coefficient_approximator


@pytest.mark.parametrize("test_input,expected", [
	# (
	# 	(omega, sigma_n, tau, v_f, T, T_c, c_light),
	# 	(xi, nu, t, A, B)
	# )
	(
			(1e9, 1e16, 1e-14, 2e6, 0.8e11, 1e11, 3e8),
			(0.007307411691175783, 730.7411691175784, 0.5845929352940626, 0.00004871607794117188, 10000000)
	)
])
def test_dedimensionalise_parameters(test_input, expected):

	actual_parameters = pynam.dielectric.nam_dielectric_coefficient_approximator.get_dedimensionalised_parameters(*test_input)

	np.testing.assert_almost_equal(
		actual_parameters.xi, expected[0],
		decimal=6, err_msg='xi incorrectly calculated'
	)

	np.testing.assert_almost_equal(
		actual_parameters.nu, expected[1],
		decimal=6, err_msg='nu incorrectly calculated'
	)
	np.testing.assert_almost_equal(
		actual_parameters.t, expected[2],
		decimal=6, err_msg='t incorrectly calculated'
	)
	np.testing.assert_almost_equal(
		actual_parameters.a, expected[3],
		decimal=6, err_msg='A incorrectly calculated'
	)
	np.testing.assert_almost_equal(
		actual_parameters.b, expected[4],
		decimal=6, err_msg='B incorrectly calculated'
	)


@pytest.mark.parametrize("test_input,expected", [
	# (
	# 	(omega, sigma_n, tau, v_f, T, T_c, c_light),
	# 	(a, b, c, d, u_l)
	# )
	(
		(1e9, 1e16, 1e-14, 2e6, 0.8e11, 1e11, 3e8),
	 	(3.789672906817707e10, 3.257134605133221e8, 2.655709897616547e18, 2.15e16, 7.007759408279888e7)
	)
])
def test_nam_coefficients(test_input, expected):

	actual_coefficients = pynam.dielectric.nam_dielectric_coefficient_approximator.get_nam_dielectric_coefficients(*test_input)

	np.testing.assert_allclose(
		actual_coefficients.a, expected[0],
		rtol=1e-6, err_msg='a incorrectly calculated'
	)

	np.testing.assert_allclose(
		actual_coefficients.b, expected[1],
		rtol=1e-6, err_msg='b incorrectly calculated'
	)
	np.testing.assert_allclose(
		actual_coefficients.c, expected[2],
		rtol=1e-2, err_msg='c incorrectly calculated'
	)
	np.testing.assert_allclose(
		actual_coefficients.d, expected[3],
		rtol=1e-2, err_msg='d incorrectly calculated'
	)
	np.testing.assert_allclose(
		actual_coefficients.u_l, expected[4],
		rtol=1e-5, err_msg='u_l incorrectly calculated'
	)


def test_nam_eps():
	u_c = 1e15
	eps_to_test = pynam.dielectric.nam_dielectric_coefficient_approximator.get_nam_dielectric_coefficients(
		1e9,
		1e16,
		1e-14,
		2e6,
		0.8e11,
		1e11,
		3e8).eps()

	np.testing.assert_allclose(
		eps_to_test(10, u_c), -3.789672906817707e10 + 3.257134605133221e8j,
		rtol=1e-3, err_msg='below u_l bad'
	)

	np.testing.assert_allclose(
		eps_to_test(1e10, u_c), -2.655709887616547e8 + 2.302290450767144e6j,
		rtol=1e-3, err_msg='linear region bad'
	)

	np.testing.assert_allclose(
		eps_to_test(1e17, u_c), 1,
		rtol=1e-6, err_msg='above cutoff bad'
	)
