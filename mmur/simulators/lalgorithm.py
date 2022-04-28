import warnings

import numpy as np
import xgboost as xgb
from joblib import Parallel, delayed
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from mmur.generators.blob_generator import BlobGenerator


def _predict_label(model, X_test, tau):
    """Outputs the labels given by a trained model applied to a set of feature observations, with a specified classification threshold

    Parameters
    ----------
    model : _type_
        _description_
    X_test : _type_
        _description_
    tau : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    probas = model.predict_proba(X_test)
    y_pred = (probas[:, 1] > tau).astype(int)
    return y_pred


def _unpack_data_dict(data_dict, stack=False):
    """Unpacks a data dictionary into X and y, train and test variables

    Parameters
    ----------
    data_dict : _type_
        _description_
    stack : bool, optional
        _description_, by default False

    Returns
    -------
    _type_
        _description_
    """
    X_train = data_dict['train']['X']
    y_train = data_dict['train']['y']
    X_test = data_dict['test']['X']
    y_test = data_dict['test']['y']
    if stack:
        X = np.hstack([X_train, X_test])
        y = np.hstack([y_train, y_test])
        return X, y
    return X_train, X_test, y_train, y_test


def _fit_model(model, X_train, y_train, shuffle=False, val_frac=None, random_state=None):
    generator = np.random.default_rng(random_state)
    sklearn_generator = np.random.RandomState(
        generator.integers(low=0, high=np.iinfo(np.uint32).max))  # TODO: seeding

    if shuffle:
        X_train, y_train = shuffle(
            X_train, y_train, random_state=sklearn_generator)

    if str(type(model)) == "<class 'sklearn.neural_network._multilayer_perceptron.MLPClassifier'>":
        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            try:
                model.fit(X_train, y_train)
            except Warning:
                print('Model did not converge')
                return False

    # TODO: isinstance(object) (google)
    elif str(type(model)) == "<class 'xgboost.sklearn.XGBClassifier'>" and val_frac is not None:
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=int(len(y_train)*val_frac), random_state=sklearn_generator)
        model.fit(X_train, y_train, eval_set=[
            (X_val, y_val)], early_stopping_rounds=5)

    else:
        model.fit(X_train, y_train)
    return model


def execute_model(params, tau, seed, fixed_data=False, data_seed=123):
    """Implements a learning algorithm on a train/test sample from the data generator, resulting in a confusion matrix

        Parameters
        ----------
        params : dict
            Contains the parameters used to determine the learning algorithm configuration
                'data_generator' : contains the class for data generation
                'data_kwargs' : contains all data parameters
                'model' : contains the model class (sklearn implementation)
                'init_kwargs' : contains a dict of settings for the model initialization
                'fit_kwargs' : contains a dict of settings for the model fitting process
        tau : float
            The classification threshold
        seed : int
            Seed to be turned into a generator
        fixed_data : bool
            Whether to use a fixed train and test set, default is False
        data_seed : int
            If using a fixed train and set set, the seed for that sample

        Returns
        -------
        np.ndarray
            The confusion matrix resulting from training the model and applying it to the test set, of shape [4,]
        """
    rng = np.random.default_rng(seed)
    init_seed = rng.integers(low=0, high=np.iinfo(
        np.uint32).max)  # TODO: seeding

    # Initialize a model instance
    init_model = params["model"](
        random_state=init_seed, **params['init_kwargs'])

    # Create a data generator instance
    data_generator = params['data_generator'](**params['data_kwargs'])

    # Generate the data sample
    if fixed_data is True:
        data_dict = data_generator.create_train_test(random_state=data_seed)
    else:
        data_dict = data_generator.create_train_test(random_state=rng)
    X_train, X_test, y_train, y_test = _unpack_data_dict(data_dict)

    # If a label does not appear in the train or test set, ignore the run
    if len(np.unique(y_train)) != 2 or len(np.unique(y_test)) != 2:
        return np.repeat(np.nan, 4)

    # Fit model to data
    fitted_model = _fit_model(
        model=init_model, X_train=X_train, y_train=y_train, random_state=rng, **params['fit_kwargs'])

    if not fitted_model:  # In case the model did not converge, ignore the run
        return np.repeat(np.nan, 4)

    # Predict labels
    predictions = _predict_label(fitted_model, X_test, tau)

    return confusion_matrix(y_test, predictions, labels=[0, 1]).flatten()


class LAlgorithm():
    """
    Applies a learning algorithm to data generated by the generator

    Parameters
    -------
    model_name : {'LR', 'NN', 'XGB'}
        The type of algorithm to be used. Logistic Regression (LR), Neural Network (NN) and XGBoost (XGB) 
        are currently supported. The first two use Sklearn's implementation, while the xgboost package is used for XGBoost.

    data_generator : obj
        The data generator or DGP used to sample data from. Currently only BlobGenerator is supported.

    random_state : int, RandomState instance or None, default = None
        Seed to allow for reproducible results.

    kwargs : dict[str, dict]
        Contains 3 dictionaries with parameters that configure the data generator and the learning algorithm
            * 'data_kwargs' contains the parameters to configure the data generator
            * 'init_kwargs' contains the parameters that specify the the initialization of the model instance. Contains for example, the number of hidden nodes for a neural network.
            * 'fit_kwargs' contains settings used during the fitting of the model to the data. This contains for example the size of the validation set used for early stopping with XGBoost.
    """

    def __init__(
        self,
        model_name,
        data_generator,
        random_state=None,
        **kwargs
    ):

        model_names = {'LR', 'NN', 'XGB'}

        self.model_name = model_name
        self.data_generator = data_generator
        self.data_kwargs = kwargs.get('data_kwargs')
        self.init_kwargs = kwargs.get('init_kwargs')
        self.fit_kwargs = kwargs.get('fit_kwargs')

        self.rng = np.random.default_rng(random_state)

        # Define model class and store model parameters in self.model_kwargs
        self.model_kwargs = {}
        self.fit_kwargs = {}
        if self.model_name == 'LR':
            self.model = LogisticRegression
        elif self.model_name == 'NN':
            self.model = MLPClassifier
        elif self.model_name == 'XGB':
            self.model = xgb.XGBClassifier
        else:
            raise NameError('Model name not one of ', model_names)

    def _create_model_kwargs(self):
        kwargs = {'data_generator': self.data_generator, 'data_kwargs': self.data_kwargs, 'model': self.model,
                  'init_kwargs': self.init_kwargs, 'fit_kwargs': self.fit_kwargs}
        return kwargs

    def sim_true_cms(self, n_runs, tau=0.5, n_jobs=None, seed=None, fixed_data=False, data_seed=None):
        """
        Simulates the "true" holdout set distribution of the confusion matrix.  
        A train and test set are generated, and the specified model is applied
        independently a specified number of times.

        Parameters
        ----------
        n_runs : int
            the number iterations, equal to the number of train/test samples to generate

        tau : float, default = 0.5
            Classification threshold that translates a continous model output to a integer value, the predicted label

        n_jobs : int, default = None
            number of CPU's used to perform computations in parallel (using joblib's 'Parallel' 
            and 'delayed' functions)

        seed : int, default = None
            for reproducible results

        Returns
        -------
        cms : np.ndarray
            of shape[n_runs,4], contains the confusion matrices.
        """

        generator = np.random.default_rng(seed)

        # Generate seeds to give to each thread
        thread_seeds = generator.integers(low=0, high=np.iinfo(
            np.uint64).max, size=n_runs, dtype=np.uint64)

        kwargs = self._create_model_kwargs()
        cms = np.vstack(Parallel(n_jobs=n_jobs, verbose=0)(
            delayed(execute_model)(kwargs, tau, seed=thread_seeds[i], fixed_data=fixed_data, data_seed=data_seed) for i in range(n_runs)))
        cms = cms[~np.isnan(cms).any(axis=1)].astype(int)

        self.true_cms = cms  # save confusion matrices
        return cms

    def repeat_nd_train(self, n_runs, tau=0.5, seed=None, data_seed=None, n_jobs=None):
        """
        Repeats training and testing for the same data sample and split. Used to evaluate variations due to nondeterministic training

        Parameters
        -------
        n_runs : int, 
            the number of iterations of training and testing an algorithm on the same dataset and train/test split

        tau : float, default = 0.5
            the classification threshold, if the model output is higher than this value, the positive class (with label=1) is predicted, otherwise
            the negative class is predicted

        seed : int, default = None
            seeding used for training the model

        data_seed : int, default = None 
            if specified, a fixed dataset is guaranteed

        n_jobs : int, default = None 
            number of CPU's used to perform computations in parallel (using joblib's 'Parallel' 
            and 'delayed' functions)

        Returns
        -------
        cms : np.ndarray of shape (n_runs,4)
            the resulting confusion matrices corresponding to each run
        """

        generator = np.random.default_rng(seed)

        # Generate seeds to give to each thread
        thread_seeds = generator.integers(low=0, high=np.iinfo(
            np.uint64).max, size=n_runs, dtype=np.uint64)

        kwargs = self._create_model_kwargs()
        cms = np.vstack(Parallel(n_jobs=n_jobs, verbose=0)(
            delayed(execute_model)(kwargs, tau, seed=thread_seeds[i], fixed_data=True, data_seed=data_seed) for i in range(n_runs)))

        # remove failed runs
        cms = cms[~np.isnan(cms).any(axis=1)].astype(int)
        return cms


if __name__ == '__main__':
    data_generator = BlobGenerator

    data_kwargs = {'train_size': 5000,
                   'test_size': 1000,
                   'weights': [0.8, 0.2]}

    init_kwargs = {}
    fit_kwargs = {}

    kwargs = {'data_kwargs': data_kwargs,
              'init_kwargs': init_kwargs, 'fit_kwargs': fit_kwargs}

    LA = LAlgorithm('LR', data_generator, **kwargs)
    cms = LA.sim_true_cms(10, 0.5, n_jobs=-2, seed=1)

    print(cms)
