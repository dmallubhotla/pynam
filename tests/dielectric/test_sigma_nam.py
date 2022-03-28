import pynam.dielectric.sigma_nam
import numpy as np
import pytest


@pytest.mark.parametrize(
	"test_input,expected",
	[((1, 2), 7 / (2 * np.sqrt(6))), ((2, 0.25), -5j / np.sqrt(39))],
)
def test_g(test_input, expected):
	np.testing.assert_almost_equal(
		pynam.dielectric.sigma_nam.g(*test_input),
		expected,
		decimal=7,
		err_msg="g function is off",
	)


@pytest.mark.parametrize(
	"test_input,expected", [((1, 2, 0), 2), ((1, 2, 3), 2 - 3j), ((1, 0, 3), -3j)]
)
def test_s(test_input, expected):
	np.testing.assert_almost_equal(
		pynam.dielectric.sigma_nam.s(*test_input),
		expected,
		decimal=7,
		err_msg="s function is off",
	)


@pytest.mark.parametrize(
	"test_input,expected",
	[
		((1, 2, 0), 4 - 3 * np.log(3)),
		((1, 1, 0.1), 1.7258022209219421 + 0.4146045220413866j),
		((1, 2, 1), 0.535971651563 + 0.291580606867j),
		((1, 0, 1), (np.pi - 2) * 1j),
		((2, 1.09637631718, 0), 0.97892512273 + 1.09875591859j),
		((2, 1, 0), 0.91197960825 + 1.17809724510j),
	],
)
def test_f(test_input, expected):
	np.testing.assert_almost_equal(
		pynam.dielectric.sigma_nam.f(*test_input),
		expected,
		decimal=7,
		err_msg="f function is off",
	)


@pytest.mark.parametrize(
	"test_input,expected",
	[
		((2, 3, 1, 0), 1.43292419807),
		((1, 2, 3, 4), 0.020963572915 + 0.441546735048j),
		(
			(1, 2, 2, 0),
			2.24702466263660598724031013350450660504
			+ 2.6687342075059525776833106878900822165j,
		),
	],
)
def test_i1(test_input, expected):
	np.testing.assert_almost_equal(
		pynam.dielectric.sigma_nam.i1(*test_input),
		expected,
		decimal=7,
		err_msg="i1 function is off",
	)


@pytest.mark.parametrize(
	"test_input,expected",
	[
		((2, 3, 1, 0), 1.48649022993),
		((1, 2, 3, 4), 0.079899419983 + 0.441546735048j),
		(
			(1, 2, 2, 0),
			2.5083371377336783093007366990156440969
			+ 2.6687342075059525776833106878900822165j,
		),
	],
)
def test_i2(test_input, expected):
	np.testing.assert_almost_equal(
		pynam.dielectric.sigma_nam.i2(*test_input),
		expected,
		decimal=7,
		err_msg="i1 function is off",
	)


@pytest.mark.parametrize(
	"test_input,expected",
	[
		((1, 2, 3, 4), 0.228396),
		(
			(0.007307411691175783, 1e8, 730.7411691175784, 0.5845929352940626),
			1.37272e-7,
		),
	],
)
def test_a(test_input, expected):
	actual = np.real_if_close(pynam.dielectric.sigma_nam.a(*test_input))
	np.testing.assert_allclose(actual, expected, rtol=1e-5, err_msg="a function is off")


@pytest.mark.parametrize(
	"test_input,expected",
	[
		(
			(2, 1, 2, 0, 4),
			0.19089933550122580816258795500979108668
			+ 0.30273783507819906415704926284048453889j,
		),
		((100, 1, 2, 3, 4), -2.62529549976e-6 + 2.60765e-12j),
	],
)
def test_b_int(test_input, expected):
	actual = np.real_if_close(pynam.dielectric.sigma_nam.b_int(*test_input))
	np.testing.assert_almost_equal(
		actual, expected, decimal=6, err_msg="b int function is off"
	)


@pytest.mark.parametrize(
	"test_input,expected",
	[
		((1, 2, 0, 4), 1.23149 + 2.08627j),
		((1, 2, 3, 4), -0.0595819 + 0.437385j),
	],
)
def test_b(test_input, expected):
	actual = np.real_if_close(pynam.dielectric.sigma_nam.b(*test_input))
	np.testing.assert_almost_equal(
		actual, expected, decimal=6, err_msg="b function is off"
	)


@pytest.mark.parametrize(
	"test_input,expected",
	[
		((1, 2, 0, 4), 0),
		((1, 2, 3, 4), 0.984117 + 0.647951j),
		(
			(0.007307411691175783, 1e8, 730.7411691175784, 0.5845929352940626),
			0.00008925294700016892 + 0.0102953966846717j,
		),
	],
)
def test_sigma_nam(test_input, expected):
	actual = np.real_if_close(pynam.dielectric.sigma_nam.sigma_nam(*test_input))
	np.testing.assert_allclose(
		actual, expected, rtol=1e-3, err_msg="sigma_nam function is off"
	)


def test_sigma_nam_benchmark(benchmark):
	result = benchmark(pynam.dielectric.sigma_nam.sigma_nam, 1, 2, 3, 4)
	np.testing.assert_almost_equal(
		result,
		0.984117 + 0.647951j,
		decimal=6,
		err_msg="sigma nam benchmrak function is off",
	)
