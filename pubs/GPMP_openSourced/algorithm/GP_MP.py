'''
Gaussian process movement primitives
Please find the corresponding article for more details about the algorithm
'''
import autograd.numpy as np
from autograd import value_and_grad
from scipy.optimize import minimize
import autograd.scipy.stats.multivariate_normal as mvn
from autograd.numpy.linalg import solve
from scipy.optimize import NonlinearConstraint
from scipy.optimize import BFGS


np.random.seed(3)


class GpMp:
    def __init__(self, X, y, X_, y_, observation_noise=0.1):
        '''
        :param X: Original input set
        :param y: Original output set
        :param X_: Via-points input set
        :param y_: Via-points output set
        :param observation_noise: Observation noise for y
        '''
        self.X_total = np.vstack((X, X_))
        self.y_total = np.vstack((y.reshape(-1, 1), y_.reshape(-1, 1))).reshape(-1)
        self.X = X
        self.y = y
        self.X_ = X_
        self.y_ = y_
        self.input_dim = np.shape(self.X_total)[1]
        self.input_num = np.shape(self.X_total)[0]
        self.via_points_num = np.shape(X_)[0]
        self.observation_noise = observation_noise

        # Initialize the parameters
        self.param = self.init_random_param()

    def init_random_param(self):
        '''
        Initialize the hyper-parameters of GP-MP
        :return: Initial hyper-parameters
        '''
        kern_length_scale = 0.1 * np.random.normal(size=self.input_dim) + 1
        kern_noise = 1 * np.random.normal(size=1)
        return np.hstack((kern_noise, kern_length_scale))

    def build_objective(self, param):
        '''
        Compute the objective function (log pdf)
        :param param: Hyper-parameters of GP-MP
        :return: Value of the obj function
        '''
        cov_y_y_total = self.rbf(self.X_total, self.X_total, param)
        variance_matrix = np.zeros((self.input_num, self.input_num)) * 1.0
        variance_matrix[0:(self.input_num - self.via_points_num), 0:(self.input_num - self.via_points_num)] = \
            self.observation_noise**2 * np.eye(self.input_num - self.via_points_num)
        cov_y_y_total = cov_y_y_total + variance_matrix
        out = - mvn.logpdf(self.y_total, np.zeros(self.input_num), cov_y_y_total)
        return out

    def train(self):
        def cons_f(param):
            '''
            Constrained function, see Eq.(20) of the article
            :param param: Hyper-parameters of GP-MP
            :return: Value of the constrained function
            '''
            delta = 1e-10
            cov_y_y_ = self.rbf(self.X_, self.X_, param)
            min_eigen = np.min(np.linalg.eigvals(cov_y_y_))
            return min_eigen - delta

        # Using "trust-constr" approach to minimize the obj
        nonlinear_constraint = NonlinearConstraint(cons_f, 0.0, np.inf, jac='2-point', hess=BFGS())
        result = minimize(value_and_grad(self.build_objective), self.param, method='trust-constr', jac=True,
                          options={'disp': True, 'maxiter': 500, 'xtol': 1e-50, 'gtol': 1e-20},
                          constraints=[nonlinear_constraint], callback=self.callback)

        # Pre-computation for prediction
        self.param = result.x
        variance_matrix = np.zeros((self.input_num, self.input_num)) * 1.0
        variance_matrix[0:(self.input_num - self.via_points_num), 0:(self.input_num - self.via_points_num)] = \
            self.observation_noise ** 2 * np.eye(self.input_num - self.via_points_num)
        self.cov_y_y_total = self.rbf(self.X_total, self.X_total, self.param) + variance_matrix
        self.beta = solve(self.cov_y_y_total, self.y_total)
        self.inv_cov_y_y_total = solve(self.cov_y_y_total, np.eye(self.input_num))

    def rbf(self, x, x_, param):
        '''
        Interface to compute the Variance matrix (vector) of GP,
        :param x: Input 1
        :param x_: Input 2
        :param param: Hyper-parameters of GP-MP
        :return: Variance matrix (vector)
        '''
        kern_noise = param[0]
        sqrt_kern_length_scale = param[1:]
        diffs = np.expand_dims(x / sqrt_kern_length_scale, 1) - np.expand_dims(x_ / sqrt_kern_length_scale, 0)
        return kern_noise**2 * np.exp(-0.5 * np.sum(diffs ** 2, axis=2))

    def predict_determined_input(self, x):
        '''
        Compute the mean and variance functions of the posterior estimation
        :param x: Query inputs
        :return: Mean and variance functions
        '''
        cov_y_f = self.rbf(self.X_total, x, self.param)
        mean_outputs = np.dot(cov_y_f.T, self.beta.reshape((-1, 1)))
        var = (self.param[0]**2 - np.diag(np.dot(np.dot(cov_y_f.T, self.inv_cov_y_y_total), cov_y_f))).reshape(-1, 1)
        return mean_outputs, var

    def callback(self, param, state):
        # ToDo: add something you want to know about the training process
        if state.nit % 100 == 0 or state.nit == 1:
            print('---------------------------------- iter ', state.nit, '----------------------------------')
            print('running time: ', state.execution_time)
            print('obj_cost: ', state.fun)
            print('maximum constr_violation: ', state.constr_violation)


class BlendedGpMp:
    def __init__(self, gpmp_list):
        '''
        :param gpmp_list: trained gpmp_list
        '''
        self.num_gpmp = len(gpmp_list)
        self.gpmp_list = gpmp_list

    def predict_determined_input(self, inputs, alpha_list):
        '''
        :param inputs: a (input_num, d_input) matrix
        :param alpha_list: a (num_gpmp, input_num) matrix
        :return: Mean and variance functions
        '''
        num_input = np.shape(inputs)[0]
        var = np.empty(num_input)
        mu = np.empty(num_input)
        minimum_var = np.ones(num_input) * 1e-100
        for i in range(self.num_gpmp):
            gpmp = self.gpmp_list[i]
            mu_i, var_i = gpmp.predict_determined_input(inputs)
            mu_i = mu_i.reshape(-1)
            var_i = var_i.reshape(-1)
            var_i = np.max([minimum_var, var_i], 0)
            var = var + alpha_list[i, :] / var_i
            mu = mu + alpha_list[i, :] / var_i * mu_i
        var = 1 / var
        mu = var * mu
        return mu, var

    def predict_single_determined_input(self, input, alpha_pair):
        '''
        :param input: Single input, a (d_input,) array
        :param alpha_pair: Values of alphas of GP-MPs, a (num_gpmp,) array
        :return: Mean and variance functions
        '''
        mu_list = np.empty(self.num_gpmp)
        var_list = np.empty(self.num_gpmp)
        for i in range(self.num_gpmp):
            gpmp = self.gpmp_list[i]
            mu_i, var_i = gpmp.predict_determined_input(input.reshape(-1, 1))
            mu_list[i] = mu_i[0, 0]
            var_list[i] = var_i[0, 0]
        Matrix = np.empty((self.num_gpmp, self.num_gpmp - 1))
        for i in range(self.num_gpmp):
                Matrix[i, :] = np.delete(var_list, i, 0)
        temp = np.cumprod(Matrix, axis=1)[:, -1]
        den = np.sum(alpha_pair * temp)
        num_var = np.cumprod(var_list)[-1]
        var = num_var / den
        num_mu = np.sum(alpha_pair * temp * mu_list)
        mu = num_mu / den
        return mu, var
