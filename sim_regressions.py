import numpy as np
from sklearn.linear_model import Ridge
from scipy.special import expit
import collections
import csv

class Gaussian:
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

    def __call__(self, x):
        return np.exp(-(x-self.mu)**2/2/self.sigma**2)


class Sigmoid:
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

    def __call__(self, x):
        return expit((x-self.mu) / self.sigma)


class Regression():
    """
    1D regression model with Gaussian basis functions.
    """

    def __init__(self, x, t, centres, width, regularization_weight=1e-6, sigmoids=False):
        """
        :param x: samples of an independent variable
        :param t: corresponding samples of a dependent variable
        :param centres: a vector of Gaussian centres (should have similar range of values as x)
        :param width: sigma parameter of Gaussians
        :param regularization_weight: regularization strength parameter
        """
        if sigmoids:
            self.basis_functions = [Sigmoid(centre, width) for centre in centres]
        else:
            self.basis_functions = [Gaussian(centre, width) for centre in centres]
        self.ridge = Ridge(alpha=regularization_weight, fit_intercept=False)
        self.ridge.fit(self._get_features(x), t)

    def eval(self, x):
        """
        :param x: a new (or multiple samples) of the independent variable
        :return: the value of the curve at x
        """
        return self.ridge.predict(self._get_features(x))

    def _get_features(self, x):
        if not isinstance(x, collections.abc.Sized):
            x = [x]

        phi = np.zeros((len(x), len(self.basis_functions)))
        for i, basis_function in enumerate(self.basis_functions):
            phi[:,i] = basis_function(x)
        return phi

def get_angle_regression():
    """
    :return: Regression object for ankle angle during dorsiflexion
    """
    natural_angle_trajectory = []
    with open('natural_angle_trajectory.txt') as csv_file:
        csv_read = csv.reader(csv_file, delimiter=',')
        for row in csv_read:
            row_content = [float(row[0]), float(row[1])]
            natural_angle_trajectory.append(row_content)
    natural_angle_trajectory = np.array(natural_angle_trajectory)

    time = natural_angle_trajectory[:,0]
    angle = natural_angle_trajectory[:,1]

    centres = np.arange(0, 0.4, .075)
    width = .075
    result = Regression(time, angle, centres, width, .075, sigmoids=False)

    return result

def get_angular_velocity_regression():
    """
    :return: Regression object for angular velocity of ankle joint during dorsiflexion
    """
    time = np.arange(0, 0.4, 0.01)
    angles = get_angle_regression()
    angle_traj = np.zeros(len(time))
    for i in range(len(time)):
        angle_traj[i] = angles.eval(time[i])

    dy = np.zeros(angle_traj.shape, np.float)
    dy[0:-1] = np.diff(angle_traj) / np.diff(time)
    dy[-1] = (angle_traj[-1] - angle_traj[-2]) / (time[-1] - time[-2])
    dy = np.array(dy)

    angular_velocity = dy

    centres = np.arange(0, 0.4, .075)
    width = .075
    result = Regression(time, angular_velocity, centres, width, .075, sigmoids=False)

    return result

def get_excitation_regression(data):
    """
    :param data: Excitation-time dataset
    :return: Regression object for TA excitation
    """
    time = data[:,0]
    excitation = data[:,1]

    centres = np.arange(0, 0.4, .075)
    width = .075
    result = Regression(time, excitation, centres, width, .075, sigmoids=True)

    return result

def get_excitation_validation_reg(data):
    """
    :param data: Excitation-time dataset
    :return: Regression object for TA excitation
    """
    time = data[:,0]
    excitation = data[:,1]

    centres = np.arange(0, 0.4, .05)
    width = .05
    result = Regression(time, excitation, centres, width, .1, sigmoids=True)

    return result