import warnings

import numpy as np
import pandas as pd

from lmfit import models
from pandas import DataFrame
from scipy.signal import find_peaks
from typing import Any, Tuple, Union

from .useful_functions import is_iterable


def get_model(model_name, model_prefix=''):
    if model_name == 'voigt':
        mdl = models.VoigtModel(prefix=model_prefix)
    elif model_name == 'gauss':
        mdl = models.GaussianModel(prefix=model_prefix)
    elif model_name == 'fano':
        mdl = models.BreitWignerModel(prefix=model_prefix)
        mdl.set_param_hint('amplitude', min=0, max=np.inf)
    elif model_name == 'constant':
        mdl = models.ConstantModel(prefix=model_prefix)
    elif model_name == 'linear':
        mdl = models.LinearModel(prefix=model_prefix)
    elif model_name == 'quadratic':
        mdl = models.QuadraticModel(prefix=model_prefix)
    elif model_name.startswith('polynomial'):
        try:
            degree = int(model_name.replace('polynomial', ''))
            mdl = models.PolynomialModel(prefix=model_prefix, degree=degree)
        except ValueError:
            mdl = models.PolynomialModel(prefix=model_prefix)
    elif model_name == 'exp':
        mdl = models.ExponentialModel(prefix=model_prefix)
    elif model_name == 'exp_gauss':
        mdl = models.ExponentialGaussianModel(prefix=model_prefix)
        mdl.set_param_hint('decay', expr='1/%sgamma' % mdl.prefix)
    elif model_name == 'logistic':
        mdl = models.StepModel(prefix=model_prefix, form='logistic')
    elif model_name == 'sine':
        mdl = models.SineModel(prefix=model_prefix)
    else:
        raise ValueError('Model name not recognized.')

    return mdl


class FittingManager:

    def __init__(self, x_data, y_data, models_df,
                 input_parameters=DataFrame({'names': [], 'initial_values': [], 'is_fixed': [], 'bounds': []}),
                 weights=None, iter_callable=None, recursions=1, recursion_callable=None, options=''):
        """
        Fitting manager uses lmfit models. Currently supported models are: voigt, gauss, constant, linear, exp, logistic
        x_data: iterable list.
                independent variable data
        y_data: iterable list.
                data to be fitted
        models_df: pandas.DataFrame
                   with columns: 'names' (mandatory), 'prefixes', 'guess_index_regions'
        input_parameters: pandas.Dataframe
                          with columns: 'names' (mandatory) for variable names, 'initial_values' (mandatory) for initial
                          values, 'is_fixed' in case you want them fixed, 'bounds', in case you want the fit to bound
                          them.
        weights: iterable list
                 for weighted fits. If none, all weights are equal.
        iter_callable: function
                       Callback function to call at each iteration (default is None). It should have the signature:
                            iter_cb(params, iter, resid, *args, **kws),
                       where params will have the current parameter values, iter the iteration number, resid the current
                       residual array, and *args and **kws as passed to the objective function. (According to lmfit)
        recursions: integer
                    How many times the fit is going to run.
        recursion_callable: function
                            Callback function to call before each recursion (different than iteration). Recursion is
                            external to the lmfit package. Takes FittingManager type argument. Must return a dictionary
                            with optional keys:
                                'x_data', 'y_data', 'pars', 'weights, 'iter_cb'
                            The keys' values will be used to call the fitting function of lmfit.
        options: string
                 Defaults to ''. Accepted values for now: 'TurtonPoison'. Options overwrite recursion callable

        """

        # making data into np.ndarrays
        self._convert_data(x_data, y_data)
        self.options = options
        self._get_weights(weights)
        self.iteration_callable = iter_callable
        self.recursions = recursions
        if options == 'TurtonPoison':
            self.recursion_callable = turton_recursion
        else:
            self.recursion_callable = recursion_callable

        # getting information from models_df, i.e. model names, model prefixes and initial guessing regions
        self._retrieve_models_df_info(models_df)
        # define model list
        self.models = self.get_model_list()

        self.input_parameters = input_parameters

        # must get total model after you guess parameters
        self.init_pars = self.get_guessed_or_user_given_initial_parameters()
        self.total_model = self.get_total_model()

        self._try_fitting()

    def _convert_data(self, x, y):
        self.x_data = np.array(x)
        self.y_data = np.array(y)

    def _get_weights(self, w):
        if w is not None:
            if len(w) != len(self.y_data):
                warnings.warn('length of weights and y_data does not match. Weights are set to default.')
                w = None
        if w is None and self.options == 'TurtonPoison':
            w = get_turton_poison_weight(self.y_data)
        self.weights = w

    def _retrieve_models_df_info(self, models_df):

        if not isinstance(models_df, DataFrame):
            raise TypeError('models_df must be a dataframe of \'names\' and \'prefixes\'')
        self.model_names = np.array(models_df['names'])
        if 'prefixes' in models_df.keys():
            self.model_prefixes = np.array(models_df['prefixes'])
        else:
            self.model_prefixes = np.array(['' for name in models_df['names']])
        if 'guess_index_regions' in models_df.keys():
            self.model_guess_regions = np.array(models_df['guess_index_regions'])
        else:
            self.model_guess_regions = np.array([[0, len(self.x_data)] for name in models_df['names']])

    def get_model_list(self):
        mdls = []
        for i, model_name in enumerate(self.model_names):
            mdls.append(get_model(model_name, self.model_prefixes[i]))
        return mdls

    def get_total_model(self):
        tot_model = self.models[0]
        for mdl in self.models[1:]:
            tot_model += mdl
        return tot_model

    def get_guessed_or_user_given_initial_parameters(self):

        gir = self.model_guess_regions

        # get user defined initial values
        p0_dict = {}
        for i, par_name in enumerate(self.input_parameters['names']):
            p0_dict[par_name] = self.input_parameters['initial_values'][i]

        for i, mdl in enumerate(self.models[0:]):
            # get parameter names with the corresponding prefix, and then ditch the prefix
            mdl_prefix = mdl._prefix
            if mdl_prefix != '':
                indices_for_keys_of_interest = [par_name.startswith(mdl_prefix) for par_name in p0_dict.keys()]
                pars_of_interest = {key[len(mdl_prefix):]: p0_dict[key]
                                    for key in np.array(list(p0_dict.keys()))[indices_for_keys_of_interest]}
            else:
                pars_of_interest = p0_dict  # gives all the parameters, but lmfit ignores once that are not in the model

            # create or add to the parameters
            if i:
                pars += mdl.guess(self.y_data[gir[i][0]:gir[i][1]], x=self.x_data[gir[i][0]:gir[i][1]],
                                  **pars_of_interest)
            else:
                pars = self.models[0].guess(self.y_data[gir[0][0]:gir[0][1]], x=self.x_data[gir[0][0]:gir[0][1]],
                                            **pars_of_interest)

        # fixing voigt's stupidity of setting gamma==sigma
        voigt_indeces = np.argwhere(np.array(self.model_names) == 'voigt')
        voigt_indeces = np.reshape(voigt_indeces, -1)
        for index in voigt_indeces:
            par_str_gamma = self.model_prefixes[index] + 'gamma'
            par_str_sigma = self.model_prefixes[index] + 'sigma'
            par_str_fwhm = self.model_prefixes[index] + 'fwhm'
            pars[par_str_gamma].expr = ''
            if par_str_gamma in p0_dict:
                value = p0_dict[par_str_gamma]
            else:
                value = pars[par_str_gamma].value

            pars[par_str_gamma].set(value=value, vary=True, expr='', min=0, max=np.inf)
            # fv = 0.5346*fL + sqrt(0.2166fL^2+fG^2)
            # fG = 2*sigma*sqrt(2*log(2))
            # fl = 2*gamma
            # fv = 1.0692*gamma + sqrt(0.8664*gamma**2+5.545177444479562*sigma^2)
            fwhm_expr = '1.0692*{0}+sqrt(0.8664*{0}**2+8*log(2)*{1}**2)'.format(par_str_gamma, par_str_sigma)
            pars[par_str_fwhm].set(expr=fwhm_expr)

        # fix variables if asked by user
        if 'is_fixed' in self.input_parameters.keys():
            for i, par_name in enumerate(self.input_parameters['names']):
                if pd.notna(self.input_parameters['is_fixed'][i]):
                    pars[par_name].set(vary=not self.input_parameters['is_fixed'][i])

        # get user defined bounds
        if 'bounds' in self.input_parameters.keys():
            for i, par_name in enumerate(self.input_parameters['names']):
                if is_iterable(self.input_parameters['bounds'][i]):
                    pars[par_name].set(min=self.input_parameters['bounds'][i][0])
                    pars[par_name].set(max=self.input_parameters['bounds'][i][1])

        return pars

    def _try_fitting(self):
        try:
            self.fit_result = self.total_model.fit(self.y_data, self.init_pars, x=self.x_data, weights=self.weights,
                                                   iter_cb=self.iteration_callable)
            for i in range(self.recursions - 1):
                if self.recursion_callable is not None:
                    kwargs = self.recursion_callable(self)
                else:
                    kwargs = {}

                # Setting all the keys that are not defined
                if 'x_data' not in kwargs.keys():
                    kwargs['x_data'] = self.x_data
                if 'y_data' not in kwargs.keys():
                    kwargs['y_data'] = self.y_data
                if 'pars' not in kwargs.keys():
                    kwargs['pars'] = self.fit_result.params
                if 'weights' not in kwargs.keys():
                    kwargs['weights'] = self.weights
                if 'iter_cb' not in kwargs.keys():
                    kwargs['iter_cb'] = self.iteration_callable

                self.fit_result = self.total_model.fit(kwargs['y_data'], kwargs['pars'], x=kwargs['x_data'],
                                                       weights=kwargs['weights'], iter_cb=kwargs['iter_cb'])

            self.fit_pars = self.fit_result.params
            self.x_fit, self.y_fit = self.get_x_y_fit()
        except ValueError as e:
            warnings.warn('Fit was not possible: \n' + str(e))
            self.fit_result = None
            self.fit_pars = None
            self.x_fit = self.y_fit = None

    def retry_fitting(self):
        return self._try_fitting()

    def get_x_y_fit(self, x_min=None, x_max=None, output_points=1000):
        if x_min is None:
            x_min = np.min(self.x_data)
        if x_max is None:
            x_max = np.max(self.x_data)

        self.x_fit = np.linspace(x_min, x_max, output_points)
        self.y_fit = self.fit_result.model.eval(self.fit_result.params, x=self.x_fit)
        # self.y_fit = self.fit_result.model.eval(self.fit_result.init_params, x=self.x_fit)

        return self.x_fit, self.y_fit

    def get_x_y_init_fit(self, x_min=None, x_max=None, output_points=1000):
        if x_min is None:
            x_min = np.min(self.x_data)
        if x_max is None:
            x_max = np.max(self.x_data)

        self.x_init_fit = np.linspace(x_min, x_max, output_points)
        self.y_init_fit = self.fit_result.model.eval(self.fit_result.init_params, x=self.x_fit)

        return self.x_init_fit, self.y_init_fit

    def get_x_y_fit_components(self, x_min=None, x_max=None, output_points=1000):
        if x_min is None:
            x_min = np.min(self.x_data)
        if x_max is None:
            x_max = np.max(self.x_data)

        self.x_fit_components = np.linspace(x_min, x_max, output_points)
        self.y_fit_components = self.fit_result.model.eval_components(params=self.fit_result.params,
                                                                      x=self.x_fit_components)

        return self.x_fit_components, self.y_fit_components

    def get_x_y_init_fit_components(self, x_min=None, x_max=None, output_points=1000):
        if x_min is None:
            x_min = np.min(self.x_data)
        if x_max is None:
            x_max = np.max(self.x_data)

        self.x_init_fit_components = np.linspace(x_min, x_max, output_points)
        self.y_init_fit_components = self.fit_result.model.eval_components(params=self.fit_result.init_params,
                                                                           x=self.x_fit_components)

        return self.x_init_fit_components, self.y_init_fit_components


def linear_sine_fit(x_data, y_data, model_guess_index_regions=None,
                    input_parameters=DataFrame({'names': [], 'initial_values': [], 'is_fixed': [], 'bounds': []}),
                    weights=None):
    model_names = ['linear', 'sine']
    model_prefixes = ['', '']

    models_df = DataFrame({'names': model_names, 'prefixes': model_prefixes})
    if model_guess_index_regions is not None:
        models_df['guess_index_regions'] = model_guess_index_regions

    fitmng = FittingManager(x_data, y_data, models_df, input_parameters, weights)

    return fitmng


def quadratic_sine_fit(x_data, y_data, model_guess_index_regions=None,
                       input_parameters=DataFrame({'names': [], 'initial_values': [], 'is_fixed': [], 'bounds': []}),
                       weights=None):
    model_names = ['quadratic', 'sine']
    model_prefixes = ['', '']

    models_df = DataFrame({'names': model_names, 'prefixes': model_prefixes})
    if model_guess_index_regions is not None:
        models_df['guess_index_regions'] = model_guess_index_regions

    fitmng = FittingManager(x_data, y_data, models_df, input_parameters, weights)

    return fitmng


def voigt_linear_fit(x_data, y_data, model_guess_index_regions=None,
                     input_parameters=DataFrame({'names': [], 'initial_values': [], 'is_fixed': [], 'bounds': []}),
                     weights=None):
    model_names = ['voigt', 'linear']
    model_prefixes = ['', '']

    models_df = DataFrame({'names': model_names, 'prefixes': model_prefixes})
    if model_guess_index_regions is not None:
        models_df['guess_index_regions'] = model_guess_index_regions

    fitmng = FittingManager(x_data, y_data, models_df, input_parameters, weights)

    if fitmng.fit_result is not None:
        fwhm = fitmng.fit_result.params['fwhm']
        center = fitmng.fit_result.params['center']
    else:
        fwhm = None
        center = None

    return fitmng, fwhm, center


def voigt_linear_sine_fit(x_data, y_data, model_guess_index_regions=None,
                          input_parameters=DataFrame({'names': [], 'initial_values': [], 'is_fixed': [], 'bounds': []}),
                          weights=None):
    model_names = ['voigt', 'linear', 'sine']
    model_prefixes = ['', '', 'sine_']

    models_df = DataFrame({'names': model_names, 'prefixes': model_prefixes})
    if model_guess_index_regions is not None:
        models_df['guess_index_regions'] = model_guess_index_regions

    fitmng = FittingManager(x_data, y_data, models_df, input_parameters, weights)

    if fitmng.fit_result is not None:
        fwhm = fitmng.fit_result.params['fwhm']
        center = fitmng.fit_result.params['center']
    else:
        fwhm = None
        center = None

    return fitmng, fwhm, center


def double_voigt_linear_fit(x_data, y_data, model_guess_index_regions=None,
                            input_parameters=DataFrame({'names': [], 'initial_values': [], 'is_fixed': [],
                                                        'bounds': []}), weights=None, peaks_indices=None):
    if 'v1_center' not in list(input_parameters['names']) or 'v2_center' not in list(input_parameters['names']):
        if peaks_indices is None:
            peaks_indices, _ = find_peaks(y_data)
        if len(peaks_indices) > 1:
            peaks_indices = [item for _, item in sorted(zip(y_data[peaks_indices], peaks_indices))][
                            -2:]  # getting 2 peaks with 2 highest y.
            peaks_indices = sorted(peaks_indices)

            deeps, _ = find_peaks(-y_data)
            deeps = deeps[
                (deeps > peaks_indices[0]) & (deeps < peaks_indices[1])]  # finding all deeps in between the peaks
            deep = [item for _, item in sorted(zip(y_data[deeps], deeps))][0]  # getting 1 deep with lowest y.

            bounds_peak1 = [x_data[0], x_data[deep]]
            bounds_peak2 = [x_data[deep], x_data[-1]]

            peaks_pos = x_data[peaks_indices]

        elif len(peaks_indices) == 1:
            fitmng, fwhm, center = voigt_linear_fit(x_data, y_data, input_parameters=input_parameters, weights=weights)
            peaks_pos = [center.value - fwhm.value, center.value + fwhm.value]
            bounds_peak1 = [x_data[0], center.value]
            bounds_peak2 = [center.value, x_data[-1]]

        else:
            warnings.warn('Can not detect any peaks')
            return [None] * 5

        bounds = [bounds_peak1, bounds_peak2]
        input_parameters = input_parameters.append(DataFrame({'names': ['v1_center', 'v2_center'],
                                                              'initial_values': peaks_pos, 'bounds': bounds}),
                                                   ignore_index=True)

    model_names = ['voigt', 'voigt', 'linear']
    model_prefixes = ['v1_', 'v2_', '']
    models_df = DataFrame({'names': model_names, 'prefixes': model_prefixes})

    if model_guess_index_regions is not None:
        models_df['guess_index_regions'] = model_guess_index_regions
    # else:
    #     models_df['guess_index_regions'] = bounds + [None]

    fitmng = FittingManager(x_data, y_data, models_df, input_parameters, weights)

    if fitmng.fit_result is not None:
        fwhms = np.array([fitmng.fit_result.params['v1_fwhm'], fitmng.fit_result.params['v2_fwhm']])
        centers = np.array([fitmng.fit_result.params['v1_center'], fitmng.fit_result.params['v2_center']])
    else:
        fwhms = None
        centers = None

    return fitmng, fwhms, centers


def exponentialgaussian_linear_fit(x_data, y_data, model_guess_index_regions=None,
                                   input_parameters=DataFrame(
                                       {'names': [], 'initial_values': [], 'is_fixed': [], 'bounds': []}),
                                   recursions=3):
    model_names = ['exp_gauss', 'linear']
    model_prefixes = ['', '']

    models_df = DataFrame({'names': model_names, 'prefixes': model_prefixes})
    if model_guess_index_regions is not None:
        models_df['guess_index_regions'] = model_guess_index_regions

    fitmng = FittingManager(x_data, y_data, models_df, input_parameters, options='TurtonPoison', recursions=recursions)

    if fitmng.fit_result is not None:
        sigma = fitmng.fit_result.params['sigma']
        center = fitmng.fit_result.params['center']
        decay = fitmng.fit_result.params['decay']
    else:
        sigma = None
        center = None
        decay = None

    return fitmng, sigma, center, decay


def double_exponentialgaussian_linear_fit(x_data, y_data, model_guess_index_regions=None,
                                          input_parameters=DataFrame(
                                              {'names': [], 'initial_values': [], 'is_fixed': [], 'bounds': []}),
                                          recursions=3):
    model_names = ['exp_gauss', 'exp_gauss', 'linear']
    model_prefixes = ['eg1_', 'eg2_', '']

    models_df = DataFrame({'names': model_names, 'prefixes': model_prefixes})
    if model_guess_index_regions is not None:
        models_df['guess_index_regions'] = model_guess_index_regions

    fitmng = FittingManager(x_data, y_data, models_df, input_parameters, options='TurtonPoison', recursions=recursions)
    fitmng.init_pars['eg2_sigma'].set(expr='eg1_sigma')
    fitmng.init_pars['eg2_center'].set(expr='eg1_center')
    
    fitmng.init_pars.add('delta', max=0.99, expr='eg1_gamma/eg2_gamma')
    # fitmng.init_pars['eg2_gamma'].set(expr='eg1_gamma/delta')
    fitmng.retry_fitting()

    if fitmng.fit_result is not None:
        sigma = fitmng.fit_result.params['eg1_sigma']
        center = fitmng.fit_result.params['eg1_center']
        eg1_decay = fitmng.fit_result.params['eg1_decay']
        eg2_decay = fitmng.fit_result.params['eg2_decay']
    else:
        sigma = None
        center = None
        eg1_decay = None
        eg2_decay = None

    return fitmng, sigma, center, eg1_decay, eg2_decay


def get_turton_poison_weight(y_data):
    y_data = np.array(y_data)
    if np.min(y_data) <= 0:  # avoid dividing by zero.
        weights = 1 / (y_data + 1)
    else:
        weights = 1 / y_data

    if np.sum(weights) < 0:
        weights = 1 / (y_data + np.abs(np.min(y_data)) + 1)

    return weights


def turton_recursion(fitmng: FittingManager):
    x_data = fitmng.x_data
    pars = fitmng.fit_result.params
    weights = get_turton_poison_weight(fitmng.fit_result.model.eval(pars, x=x_data))
    dictionary = {'weights': weights}
    return dictionary


def exp_with_bg_fit_turton_poison(x_data, y_data, model_guess_index_regions=None,
                                  input_parameters=pd.DataFrame({'names': [], 'initial_values': [], 'is_fixed': [],
                                                                 'bounds': []}),
                                  recursions=3) -> FittingManager:
    pre_fitmng, const, ampl, ampl_sgn = guess_exp_with_bg_parameters(x_data, y_data)

    # getting model for actual data
    models_df = pd.DataFrame({'names': ['constant', 'exp']})

    if model_guess_index_regions is not None:
        models_df['guess_index_regions'] = model_guess_index_regions

    estimated_decay_rate = pre_fitmng.fit_pars['decay'].value
    if 'decay' not in input_parameters['names']:
        input_parameters = input_parameters.append({'names': 'decay',
                                                    'initial_values': estimated_decay_rate,
                                                    'is_fixed': False,
                                                    'bounds': [0, np.max(x_data)]},
                                                   ignore_index=True)

    if 'c' not in input_parameters['names']:
        input_parameters = input_parameters.append({'names': 'c',
                                                    'initial_values': const,
                                                    'is_fixed': False,
                                                    'bounds': [-10 * const, 10 * const]},
                                                   ignore_index=True)

    predicted_ampl_abs = ampl * pre_fitmng.fit_pars['amplitude'].value

    if 'amplitude' not in input_parameters['names']:
        input_parameters = input_parameters.append({'names': 'amplitude',
                                                    'initial_values': ampl * ampl_sgn,
                                                    'is_fixed': False,
                                                    'bounds': [-10 * predicted_ampl_abs, 10 * predicted_ampl_abs]},
                                                   ignore_index=True)

    fitmng = FittingManager(x_data, y_data, models_df,
                            input_parameters=input_parameters,
                            recursions=recursions,
                            options='TurtonPoison')

    return fitmng


def double_exp_with_bg_fit_turton_poison(x_data, y_data, model_guess_index_regions=None,
                                         input_parameters=pd.DataFrame({'names': [], 'initial_values': [],
                                                                        'is_fixed': [], 'bounds': []}),
                                         recursions=3) -> FittingManager:
    pre_fitmng, const, ampl, ampl_sgn = guess_exp_with_bg_parameters(x_data, y_data)
    fast_decay_data_length = int(np.ceil(len(x_data) / 10))
    pre_fitmng_fast, const, ampl_fast, ampl_sgn_fast = guess_exp_with_bg_parameters(x_data[:fast_decay_data_length],
                                                                                    y_data[:fast_decay_data_length])

    # getting model for actual data
    models_df = pd.DataFrame({'names': ['constant', 'exp', 'exp'],
                              'prefixes': ['', '', 'fast_']})

    if model_guess_index_regions is not None:
        models_df['guess_index_regions'] = model_guess_index_regions

    estimated_decay_rate = pre_fitmng.fit_pars['decay'].value
    if 'decay' not in input_parameters['names']:
        input_parameters = input_parameters.append({'names': 'decay',
                                                    'initial_values': estimated_decay_rate,
                                                    'is_fixed': False,
                                                    'bounds': [0, np.max(x_data)]},
                                                   ignore_index=True)

    estimated_fast_decay_rate = pre_fitmng_fast.fit_pars['decay'].value
    if 'fast_decay' not in input_parameters['names']:
        input_parameters = input_parameters.append({'names': 'fast_decay',
                                                    'initial_values': estimated_fast_decay_rate,
                                                    'is_fixed': False,
                                                    'bounds': [0, np.max(x_data[:fast_decay_data_length])]},
                                                   ignore_index=True)

    if 'c' not in input_parameters['names']:
        input_parameters = input_parameters.append({'names': 'c',
                                                    'initial_values': const,
                                                    'is_fixed': False,
                                                    'bounds': [-10 * const, 10 * const]},
                                                   ignore_index=True)

    predicted_ampl_abs = ampl * pre_fitmng.fit_pars['amplitude'].value

    if 'amplitude' not in input_parameters['names']:
        input_parameters = input_parameters.append({'names': 'amplitude',
                                                    'initial_values': ampl * ampl_sgn,
                                                    'is_fixed': False,
                                                    'bounds': [-10 * predicted_ampl_abs, 10 * predicted_ampl_abs]},
                                                   ignore_index=True)

    predicted_ampl_abs_fast = ampl_fast * pre_fitmng_fast.fit_pars['amplitude'].value

    if 'fast_amplitude' not in input_parameters['names']:
        input_parameters = input_parameters.append({'names': 'fast_amplitude',
                                                    'initial_values': ampl_fast * ampl_sgn_fast,
                                                    'is_fixed': False,
                                                    'bounds': [-10 * predicted_ampl_abs_fast,
                                                               10 * predicted_ampl_abs_fast]},
                                                   ignore_index=True)

    fitmng = FittingManager(x_data, y_data, models_df,
                            input_parameters=input_parameters,
                            recursions=recursions,
                            options='TurtonPoison')

    return fitmng


def guess_exp_with_bg_parameters(x_data, y_data) -> Tuple[
    FittingManager, Union[np.ndarray, int, float, complex], Union[Union[int, float, complex], Any], int]:
    # normalizing the exponential amplitude and removing most of background
    ampl = np.max(y_data) - np.min(y_data)
    is_ampl_positive = np.argmin(y_data) > np.argmax(y_data)
    if is_ampl_positive:
        const = np.min(y_data)
        ampl_sgn = 1
    else:
        const = np.max(y_data)
        ampl_sgn = -1

    pre_y_data = ampl_sgn * (y_data - const) / ampl
    pre_models_df = pd.DataFrame({'names': ['exp']})
    pre_input_parameters = pd.DataFrame({'names': ['decay'],
                                         'initial_values': [
                                             np.abs(1 / np.polyfit(x_data, np.log(abs(pre_y_data) + 1.e-15),
                                                                   1)[0])],
                                         'is_fixed': [False],
                                         'bounds': [[0, np.max(x_data)]]})
    pre_fitmng = FittingManager(x_data, pre_y_data, pre_models_df, input_parameters=pre_input_parameters)

    return pre_fitmng, const, ampl, ampl_sgn
