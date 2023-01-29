import os
import yaml

import cftime

from scipy.optimize import fsolve
import numpy as np
import dask
import xarray as xr

from . import gasex
from . import co2calc

path_to_here = os.path.dirname(os.path.realpath(__file__))

s_per_d = 86400.0


@dask.delayed
def sim_single_box(nday, ic_data, do_spinup=False, **init_kwargs):
    """run simulation with single_box"""

    # instantiate model
    m = box_model_simulation(
        model=single_box,
        **init_kwargs,
    )

    # optionally find a steady-state solution
    if do_spinup:
        ds_eq = m.spinup(ic_data)
        ic_data = {k: ds_eq[k] for k in ic_data.keys()}

    # run the model and return output dataset
    m.run(
        nday=nday,
        ic_data=ic_data,
    )
    return m.ds


class indexing_type(object):
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            assert isinstance(v, int)
            self.__dict__[k] = v

    def __getitem__(self, key):
        return self.__dict__[key]


class box_model_simulation(object):
    def __init__(
        self,
        model,
        forcing=None,
        calendar='noleap',
        **init_kwargs,
    ):
        """Run box model integrations.

        Parameters
        ----------

        model : obj
          Box model to integrate.

        forcing : xarray.Dataset
          Forcing data to run the model.

        calendar : string
          String describing the CF-conventions calendar. See:
            http://cfconventions.org/cf-conventions/cf-conventions#calendar
          This must match the calendar used in `forcing`.

        init_kwargs : dict, optional
          Keyword arguments to pass to `model`.

        """

        self.calendar = calendar
        self._init_forcing(forcing)
        self.obj = model(
            **init_kwargs,
        )

    def _init_forcing(self, forcing):
        """initialize forcing Dataset"""

        if forcing is None:
            self.forcing = None

        else:
            assert (
                forcing.time.encoding['calendar'] == self.calendar
            ), "forcing calendar and simulation calendar mismatch"

            self.forcing = forcing.copy()

            # determine the data_vars for interpolation
            not_data_vars = []
            tb_var = None
            if 'bounds' in forcing.time.attrs:
                tb_var = forcing.time.attrs['bounds']
                not_data_vars.append(tb_var)
            self.forcing_data_vars = list(
                filter(lambda v: v not in not_data_vars, forcing.data_vars),
            )

    def _init_state(self, ic_data):
        """initialize model state variables"""
        for v in self.obj.state_names:
            i = self.obj.sind[v]
            self.obj.state_data[i, :] = ic_data[v]

    def _init_output_arrays(self, start_date, nt):

        time, time_bnds = gen_daily_cftime_coord(
            start_date,
            nt,
            calendar=self.calendar,
        )

        # integration time set to beginning of interval
        self.time = time_bnds[:, 0].data

        self._ds = xr.Dataset()
        self._ds[time.bounds] = time_bnds
        for v, attrs in self.obj.diag_attrs.items():
            self._ds[v] = xr.DataArray(
                np.zeros((nt, self.obj.nx)),
                dims=("time", "nx"),
                attrs=attrs,
                coords={"time": time},
            )
        for v, attrs in self.obj.state_attrs.items():
            self._ds[v] = xr.DataArray(
                np.zeros((nt, self.obj.nx)),
                dims=("time", "nx"),
                attrs=attrs,
                coords={"time": time},
            )

    def _forcing_t(self, t):

        if self.forcing is None:
            return None

        interp_time = t
        if interp_time <= self.forcing.time[0]:
            return self.forcing[self.forcing_data_vars].isel(time=0)
        elif interp_time >= self.forcing.time[-1]:
            return self.forcing[self.forcing_data_vars].isel(time=-1)
        else:
            return self.forcing[self.forcing_data_vars].interp(
                time=interp_time,
                kwargs=dict(bounds_error=True),
            )

    def _post_data(self, n, state_t):
        for i, v in enumerate(self.obj.state_names):
            self._ds[v][n, :] = state_t[i, :]

        for i, v in enumerate(self.obj.diag_names):
            self._ds[v][n, :] = self.obj.diag_data[i, :]

    @property
    def ds(self):
        """Data comprising the output from ``box_model_instance``."""
        return self._ds

    def _compute_tendency(self, t, state_t, run_kwargs):
        """Return the feisty time tendency."""
        return self.obj.compute_tendencies(
            state_data=state_t,
            forcing_t_ds=self._forcing_t(t),
            **run_kwargs,
        )

    def _solve(self, nt, method, run_kwargs):
        """Call a numerical ODE solver to integrate the feisty model in time."""

        state_t = self.obj.state_data

        if method == "euler":
            self._solve_foward_euler(nt, state_t, run_kwargs)

        elif method in ["Radau", "RK45"]:
            # TODO: make input arguments
            self._solve_scipy(nt, state_t, method, run_kwargs)
        else:
            raise ValueError(f"unknown method: {method}")

    def _solve_foward_euler(self, nt, state_t, run_kwargs):
        """use forward-euler to integrate model"""
        for n in range(nt):
            dsdt = self._compute_tendency(self.time[n], state_t, run_kwargs) * s_per_d
            state_t[:, :] = state_t[:, :] + dsdt[:, :] * self.dt
            self._post_data(n, state_t)

    def _solve_scipy(self, nt, state_t, method, run_kwargs):
        """use a SciPy solver to integrate the model equation."""
        raise NotImplementedError("scipy solvers not implemented")

    def run(
        self,
        nday,
        ic_data,
        start_date='0001-01-01',
        file_out=None,
        method="euler",
        run_kwargs={},
    ):
        """Integrate the FEISTY model.

        Parameters
        ----------

        nday : integer
          Number of days to run.

        ic_data : dict_like
          Dataset or dictionary that includes data for each of the model's
          state variables.

        start_date : string, optional
          Date to start the model integration; must have format 'YYYY-MM-DD'.

        file_out : string, optional
          File name to write model output data.

        method : string
          Method of integrating model equations. Options: ['euler', 'Radau', 'RK45'].

          .. note::
             Only ``method='euler'`` is supported currently.
        """

        # time step
        self.dt = 1.0  # day
        nt = nday

        self._init_state(ic_data)
        self._init_output_arrays(start_date, nt)
        self._solve(nt, method, run_kwargs)
        self._shutdown(file_out)

    def _shutdown(self, file_out):
        """Close out integration:
        Tasks:
            - write output
        """
        if file_out is not None:
            self._ds.to_netcdf(file_out)

    def spinup(self, ic_data, nday=1, run_kwargs={}):
        """use SciPy.fsolve to find an equilibrium solution"""

        assert set(self.obj.state_names) == set(ic_data.keys())

        ntracers, nx = self.obj.state_data.shape

        def wrap_model(state_in_flat):
            state_in = state_in_flat.reshape((ntracers, nx))
            self.run(
                nday=nday,
                ic_data={k: state_in[i, :] for i, k in enumerate(self.obj.state_names)},
                run_kwargs=run_kwargs,
            )
            state_out = self.obj.state_data.ravel()

            return (state_out - state_in_flat) ** 2

        # setup dataset for output
        state_equil = xr.Dataset()
        for v, attrs in self.obj.state_attrs.items():
            state_equil[v] = xr.DataArray(
                np.zeros((self.obj.nx)),
                dims=("nx"),
                attrs=attrs,
            )

        # initial guess
        state0_flat = np.vstack([ic_data[k] for k in self.obj.state_names]).ravel()
        statef_flat = fsolve(wrap_model, state0_flat, xtol=1e-7, maxfev=2000)
        statef = statef_flat.reshape((ntracers, nx))

        for i, v in enumerate(self.obj.state_names):
            state_equil[v].data[:] = statef[i, :]

        return state_equil


class single_box(object):
    """
    A box model
    """

    def __init__(self, **kwargs):
        """Initialize model"""

        # static attributes
        self.boxes = [
            "surface",
        ]
        self.state_names = [
            "dic",
            "alk",
        ]
        self.settings = [
            'lapply_alk_flux',
            'lventilate',
            'tracer_boundary_conc',
            'diag_list',
        ]
        self.forcing_vars = [
            'salt',
            'temp',
            'fice',
            'patm',
            'u10',
            'h',
            'area',
            'Xco2atm',
            'ventilation_flow',
            'alk_flux',
        ]

        # validate kwargs
        unknown_args = set(kwargs.keys()) - set(self.settings + self.forcing_vars)
        assert not unknown_args, f'Unknown keyword argument(s): {unknown_args}'

        # initialization sequence
        self._init_state()
        self._init_diag()
        self._init_model(**kwargs)
        self._init_forcing(**kwargs)
        self._pH0 = 8.0

    def _init_state(self):
        """initialize model state"""
        with open(f"{path_to_here}/state_variables.yml") as fid:
            tracer_attrs = yaml.safe_load(fid)

        self.nx = len(self.boxes)
        self.ntracers = len(self.state_names)
        self.state_attrs = {k: tracer_attrs[k] for k in self.state_names}

    def _init_diag(self, diag_list=None):
        """initialize model diagnostics"""
        diag_attrs_files = [
            f"{path_to_here}/csys_diag_definitions.yml",
        ]

        diag_defs = {}
        for file in diag_attrs_files:
            with open(file) as fid:
                diag_defs.update(yaml.safe_load(fid))

        self.diag_attrs = {k: v['attrs'] for k, v in diag_defs.items()}
        if diag_list is not None:
            diag_defs = {k: v for k, v in diag_defs.items() if k in diag_list}
            self.diag_attrs = {k: v for k, v in self.diag_attrs.items() if k in diag_list}

        self.diag_units_convert_factor = {}
        for k, v in diag_defs.items():
            try:
                self.diag_units_convert_factor[k] = v['units_convert_factor']
            except KeyError:
                self.diag_units_convert_factor[k] = 1.0

        self.diag_names = list(self.diag_attrs.keys())
        self.ndiag = len(self.diag_names)

    def _init_model(self, **kwargs):
        """initialize memory and model settings"""

        # parse the settings_dict
        self.lapply_alk_flux = kwargs.pop('lapply_alk_flux', False)
        self.lventilate = kwargs.pop('lventilate', False)
        tracer_boundary_conc_dict = kwargs.pop('tracer_boundary_conc', None)

        # initialize arrays
        self.state_data = np.zeros((self.ntracers, self.nx))
        self.tendency_data = np.zeros((self.ntracers, self.nx))
        self.diag_data = np.zeros((self.ndiag, self.nx))
        self.tracer_boundary_conc = np.zeros((self.ntracers, self.nx))

        if self.lventilate:
            assert (
                tracer_boundary_conc_dict is not None
            ), 'lventilate=True requires `tracer_boundary_conc_dict` to be set'
            for i, v in enumerate(self.state_names):
                self.tracer_boundary_conc[i, :] = tracer_boundary_conc_dict[v]

        # initialize indexers
        self.sind = indexing_type(
            **{k: i for i, k in enumerate(self.state_names)},
        )
        self.dind = indexing_type(
            **{k: i for i, k in enumerate(self.diag_names)},
        )

    def _init_forcing(self, **kwargs):
        """initialize forcing data"""
        self.forcing_constant = {v: kwargs.pop(v, None) for v in self.forcing_vars}

    def compute_tendencies(self, state_data, forcing_t_ds=None, **kwargs):
        """compute tendencies"""

        # unpack inputs into local variables
        dic_data = state_data[self.sind.dic, :]
        alk_data = state_data[self.sind.alk, :]

        forcing_data = {}
        for v in self.forcing_vars:
            if self.forcing_constant[v] is not None:
                forcing_data[v] = self.forcing_constant[v]
            else:
                forcing_data[v] = forcing_t_ds[v].data

        salt_data = forcing_data['salt']
        temp_data = forcing_data['temp']
        fice_data = forcing_data['fice']
        patm_data = forcing_data['patm']
        u10_data = forcing_data['u10']
        Xco2atm_data = forcing_data['Xco2atm']
        h_data = forcing_data['h']
        area_data = forcing_data['area']
        alk_flux_data = forcing_data['alk_flux']
        ventilation_flow_data = forcing_data['ventilation_flow']

        vol = area_data * h_data

        # initialize tendency
        self.tendency_data[:, :] = 0.0

        # compute dic tendency terms
        co2sol = co2calc.co2sol(salt_data, temp_data)  # mmol/m^3/atm
        thermodyn = co2calc.co2_eq_const(salt_data, temp_data)

        # solve carbonate system
        co2aq, pH = co2calc.calc_csys_iter(
            dic_data, alk_data, salt_data, temp_data, pH0=self._pH0, thermodyn=thermodyn
        )

        # carbonate system diagnostics
        self.diag_data[self.dind.pH, :] = pH
        self._pH0 = pH

        self.diag_data[self.dind.pco2, :] = 1.0e6 * co2aq / co2sol

        rf, ddicdco2 = co2calc.rf_ddicdco2(
            salt_data, temp_data, dic_data, co2aq, pH, thermodyn=thermodyn
        )
        self.diag_data[self.dind.revelle_factor, :] = rf
        self.diag_data[self.dind.dDICdCO2, :] = ddicdco2

        # compute gas exchange
        k_gas = gasex.gas_transfer_velocity(u10_data, temp_data)  # m/s
        self.diag_data[self.dind.xkw, :] = k_gas

        co2atm = patm_data * (Xco2atm_data * 1.0e-6) * co2sol  # mmol/m^3
        gasex_co2 = (1.0 - fice_data) * k_gas * (co2atm - co2aq)  # mmol/m^2/s
        self.diag_data[self.dind.fgco2, :] = gasex_co2 * self.diag_units_convert_factor['fgco2']
        self.tendency_data[self.sind.dic, :] += gasex_co2 * area_data  # mmol/s

        # apply alk forcing
        if self.lapply_alk_flux:
            self.tendency_data[self.sind.alk, :] += alk_flux_data * area_data  # mmol/s

        # apply boundary flow
        if self.lventilate:
            self.tendency_data[:, :] += (
                ventilation_flow_data * self.tracer_boundary_conc
                - ventilation_flow_data * self.state_data
            )  # m^3/s * mmol/m^3 --> mmol/s

        # normalize by volume
        self.tendency_data /= vol  # mmol/m^3/s

        return self.tendency_data


def get_forcing_defaults(values_only=False):
    with open(f"{path_to_here}/forcing_variables.yml") as fid:
        forcing_var_defs = yaml.safe_load(fid)
    if values_only:
        return {k: d['default_value'] for k, d in forcing_var_defs.items()}
    else:
        return forcing_var_defs


def gen_forcing_dataset(nday, start_date='0001-01-01', calendar='noleap', **kwargs):

    time, time_bnds = gen_daily_cftime_coord(start_date, nday, calendar='noleap')

    forcing_var_defs = get_forcing_defaults()

    forcing_values = {k: v['default_value'] for k, v in forcing_var_defs.items()}
    assert not set(kwargs.keys()) - set(forcing_values.keys())

    forcing_values.update(kwargs)

    forcing = xr.Dataset()
    for v in forcing_values.keys():
        if np.isscalar(forcing_values[v]):
            data = forcing_values[v] * np.ones(nday)
        else:
            data = forcing_values[v][:]

        forcing[v] = xr.DataArray(
            data,
            dims=("time"),
            attrs=forcing_var_defs[v]['attrs'],
            coords={'time': time},
        )
    forcing[time.bounds] = time_bnds
    return forcing


def gen_daily_cftime_coord(
    start_date,
    nday,
    units='days since 0001-01-01 00:00:00',
    calendar='gregorian',
):
    time = xr.cftime_range(start=start_date, periods=nday, freq='D', calendar=calendar)

    num_time = cftime.date2num(time, units, calendar=calendar)
    time_bounds_data = np.vstack((num_time, num_time + 1)).T
    time_data = cftime.num2date(time_bounds_data.mean(axis=1), units, calendar=calendar)

    time = xr.DataArray(time_data, dims=('time'), name='time')
    time.encoding['units'] = units
    time.encoding['calendar'] = calendar
    time.encoding['dtype'] = np.float64
    time.encoding['_FillValue'] = None

    time.attrs['bounds'] = 'time_bnds'

    time_bnds = xr.DataArray(
        cftime.num2date(time_bounds_data, units, calendar),
        dims=('time', 'd2'),
        coords={'time': time},
        name='time_bnds',
    )

    time_bnds.encoding['dtype'] = np.float64
    time_bnds.encoding['_FillValue'] = None

    return time, time_bnds
