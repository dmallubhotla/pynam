import pynam.dielectric.low_k_nam
import numpy as np
import pytest


@pytest.mark.parametrize("test_input,expected", [
	((1, 2), 7 / (2 * np.sqrt(6))),
	((2, 0.25), -5j / np.sqrt(39))
])
def test_g(test_input, expected):
	np.testing.assert_almost_equal(
		pynam.dielectric.low_k_nam.g(*test_input), expected,
		decimal=7, err_msg='g function is off'
	)


@pytest.mark.parametrize("test_input,expected", [
	((1, 1, 0), 8 / 5),
	((1, 1, 1), 3 / 5 + 11j / 15),
	((1, 0, 1), 16j / 15)
])
def test_f(test_input, expected):
	np.testing.assert_almost_equal(
		pynam.dielectric.low_k_nam.f(*test_input), expected,
		decimal=7, err_msg='f function is off'
	)


@pytest.mark.parametrize("test_input,expected", [
	((2, 3, 1, 0), 1.42546694754),
	((1, 2, 3, 4), 0.0206212167 + 0.4411134527j)
])
def test_i1(test_input, expected):
	np.testing.assert_almost_equal(
		pynam.dielectric.low_k_nam.i1(*test_input), expected,
		decimal=7, err_msg='i1 function is off'
	)


@pytest.mark.parametrize("test_input,expected", [
	((2, 3, 1, 0), 1.47903168398),
	((1, 2, 3, 4), 0.079491440779 + 0.441113452718j)
])
def test_i2(test_input, expected):
	np.testing.assert_almost_equal(
		pynam.dielectric.low_k_nam.i2(*test_input), expected,
		decimal=7, err_msg='i1 function is off'
	)


@pytest.mark.parametrize("test_input,expected", [
	((1, 2, 3, 4), 0.228292),
])
def test_a(test_input, expected):
	actual = np.real_if_close(pynam.dielectric.low_k_nam.a(*test_input))
	np.testing.assert_almost_equal(
		actual, expected,
		decimal=6, err_msg='a function is off'
	)


@pytest.mark.parametrize("test_input,expected", [
	((2, 1, 2, 0, 4), 0.479529593125),
	((100, 1, 2, 3, 4), -2.62529549942e-6 + 2.60588e-12j),
])
def test_b_int(test_input, expected):
	actual = np.real_if_close(pynam.dielectric.low_k_nam.b_int(*test_input))
	np.testing.assert_almost_equal(
		actual, expected,
		decimal=6, err_msg='b int function is off'
	)


@pytest.mark.parametrize("test_input,expected", [
	((1, 2, 0, 4), 3.514889721181435),
	((1, 2, 3, 4), -0.0598057 + 0.437146j),
])
def test_b(test_input, expected):
	actual = np.real_if_close(pynam.dielectric.low_k_nam.b(*test_input))
	np.testing.assert_almost_equal(
		actual, expected,
		decimal=6, err_msg='b function is off'
	)


@pytest.mark.parametrize("test_input,expected", [
	((1, 2, 0, 4), 0),
	((1, 2, 3, 4), 0.98358 + 0.648221j),
])
def test_sigma_alk(test_input, expected):
	actual = np.real_if_close(pynam.dielectric.low_k_nam.sigma_nam_alk(*test_input))
	np.testing.assert_almost_equal(
		actual, expected,
		decimal=6, err_msg='b function is off'
	)


def test_sigma_alk_benchmark(benchmark):
	result = benchmark(pynam.dielectric.low_k_nam.sigma_nam_alk, 1, 2, 3, 4)
	np.testing.assert_almost_equal(
		result, 0.98358 + 0.648221j,
		decimal=6, err_msg='b function is off'
	)
