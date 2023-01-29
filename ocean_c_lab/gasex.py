import numpy as np


# gas exchange coefficient
xkw_coef_cm_per_hr = 0.251

# (cm/hr s^2/m^2) --> (m/s s^2/m^2)
xkw_coef = xkw_coef_cm_per_hr * 3.6e-5


def gas_transfer_velocity(u10, temp):
    """
    Compute gas transfer velocity.

    Parameters
    ----------

    u10 : numeric
      Wind speed [m/s]

    temp : numeric
      Sea surface Temperature [°C]

    Returns
    -------

    k : numeric
      Gas transfer velocity [m/s]
    """
    sc = schmidt_co2(temp)
    u10sq = u10 * u10
    return xkw_coef * u10sq * (np.sqrt(sc / 660.0))


def schmidt_co2(sst):
    """
    Compute Schmidt number of CO2 in seawater as function of SST.

    Range of validity of fit is -2:40
    Reference:
        Wanninkhof 2014, Relationship between wind speed
          and gas exchange over the ocean revisited,
          Limnol. Oceanogr.: Methods, 12,
          doi:10.4319/lom.2014.12.351

    Check value at 20°C = 668.344

    Parameters
    ----------

    sst : numeric
      Temperature

    Returns
    -------

    sc : numeric
      Schmidt number
    """
    a = 2116.8
    b = -136.25
    c = 4.7353
    d = -0.092307
    e = 0.0007555

    # enforce bounds
    sst_loc = np.where(sst < -2.0, -2.0, np.where(sst > 40.0, 40.0, sst))

    return a + sst_loc * (b + sst_loc * (c + sst_loc * (d + sst_loc * e)))
