import numpy as np

import onets_lab.gasex as gasex


def test_schmidt_co2():
    # check value
    np.testing.assert_almost_equal(gasex.schmidt_co2(20.0), 668.344)

    # values outside bounds -2.0, -40.0 should resolve to function value at bounds
    np.testing.assert_equal(
        gasex.schmidt_co2(np.array([-3.0, 42.0])),
        gasex.schmidt_co2(np.array([-2.0, 40.0])),
    )


def test_gas_transfer_velocity():
    np.testing.assert_almost_equal(
        gasex.gas_transfer_velocity(np.array([1.0, 5.0, 10.0]), 20.0),
        np.array([9.09293908e-06, 2.27323477e-04, 9.09293908e-04]),
    )
