import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split


class BlobGenerator:
    """Class for generating random 2-class classification problems.

        This is an adaptation of sklearn's 'make_classification' function. Normally distributed
        points are created for both classes, with each class having its own distribution.
        The coordinates of these point correspond to the 'informative' features. Additional redundant and repeated
        features can be added. Points coming from the same Gaussian distribution form clusters or 'blobs' and the 
        distance between these blobs is specified in terms of euclidean distance.

        Without shuffling, ``X`` horizontally stacks features in the following
        order: the primary ``n_informative`` features, followed by ``n_redundant``
        linear combinations of the informative features, followed by ``n_repeated``
        duplicates, drawn randomly with replacement from the informative and
        redundant features. The remaining features are filled with random noise.
        Thus, without shuffling, all useful features are contained in the columns
        ``X[:, :n_informative + n_redundant + n_repeated]``.

        Parameters
        ----------
        train_size : int
            number of training samples

        test_size : int
            number of testing samples

        n_features : int, default=2
            The total number of features. These comprise ``n_informative``
            informative features, ``n_redundant`` redundant features,
            ``n_repeated`` duplicated features and
            ``n_features-n_informative-n_redundant-n_repeated`` useless features
            drawn at random.

        n_informative : int, default=2
            The number of informative features. Each class is composed of a gaussian cluster. For each cluster,
            informative features are drawn independently from  N(0, 1) and then
            linearly combined within each cluster in order to add
            covariance.

        n_redundant : int, default=0
            The number of redundant features. These features are generated as
            random linear combinations of the informative features.

        n_repeated : int, default=0
            The number of duplicated features, drawn randomly from the informative
            and the redundant features.

        n_classes : int, default=2
            The number of classes (or labels) of the classification problem.

        n_clusters_per_class : int, default=1
            The number of clusters per class.

        weights : array-like of shape (n_classes,) or (n_classes - 1,)
                default=None
            The mean proportions of samples assigned to each class. If None, then
            classes are balanced. Note that if ``len(weights) == n_classes - 1``,
            then the last class weight is automatically inferred.
            More than ``n_samples`` samples may be returned if the sum of
            ``weights`` exceeds 1. Note that the actual class proportions will
            not exactly match ``weights`` when ``flip_y`` isn't 0 or random_imbalance isn't False.

        class_d : float, default=1.0
            The euclidean distance between the clusters. For now only available for two classes.

        var : float, default=1
            specifies the variance of each feature

        cov : float, default=0
        specifies the covariance between each pair of features

        flip_y : float, default=0.01
            The fraction of samples whose class is assigned randomly. Larger
            values introduce noise in the labels and make the classification
            task harder. Note that the default setting flip_y > 0 might lead
            to less than ``n_classes`` in y in some cases.

        random_imbalance : bool, default=False
            If set to True, the data sample class balance is stochastic, with the 
            number of draws for each class Multinomially distributed with 
            probabilities set to ``weights``.

        random_state : int, RandomState instance or None, default=None
            Determines random number generation for dataset creation. Pass an int
            for reproducible output across multiple function calls.
            See :term:`Glossary <random_state>`.

        **kwargs
            Other parameters. Currently consist of parameters characterizing the 
            beta distribution used as a prior in case `random_imbalance` is True

        """

    def __init__(
        self,
        train_size,
        test_size,
        n_features=2,
        n_informative=2,
        n_redundant=0,
        n_repeated=0,
        n_classes=2,
        n_clusters_per_class=1,
        weights=None,
        flip_y=0,
        class_d=2.0,
        var=1,
        cov=0,
        random_imbalance=False,
        random_state=None,
        **kwargs
    ):

        # Count features, clusters and samples
        if n_informative + n_redundant + n_repeated > n_features:
            raise ValueError(
                "Number of informative, redundant and repeated "
                "features must sum to less than the number of total"
                " features"
            )
        # Use log2 to avoid overflow errors
        if n_informative < np.log2(n_classes * n_clusters_per_class):
            msg = "n_classes({}) * n_clusters_per_class({}) must be"
            msg += " smaller or equal 2**n_informative({})={}"
            raise ValueError(
                msg.format(
                    n_classes, n_clusters_per_class, n_informative, 2 ** n_informative
                )
            )

        self.n_samples = train_size + test_size
        self.train_size = train_size
        self.test_size = test_size
        self.n_features = n_features
        self.n_informative = n_informative
        self.n_redundant = n_redundant
        self.n_repeated = n_repeated
        self.n_classes = n_classes
        self.n_clusters_per_class = n_clusters_per_class
        self.flip_y = flip_y
        self.class_d = class_d
        self.var = var
        self.cov = cov
        self.random_imbalance = random_imbalance

        if weights is not None:
            if len(weights) not in [n_classes, n_classes - 1]:
                raise ValueError(
                    "Weights specified but incompatible with number of classes."
                )

            if len(weights) == n_classes - 1:
                if isinstance(weights, list):
                    weights = weights + [1.0 - sum(weights)]
                else:
                    weights = np.resize(weights, n_classes)
                    weights[-1] = 1.0 - sum(weights[:-1])

        else:
            weights = [1.0 / n_classes] * n_classes
        self.weights = weights

        if random_imbalance:
            self.imbalance_prior_a = kwargs.get('imbalance_prior_a', None)
            self.imbalance_prior_b = kwargs.get('imbalance_prior_b', None)

        self.rng = np.random.default_rng(random_state)

    def make_classification(self, random_state=None):
        """
        Creates a sample based on the inputs provided to the instance initialization.

        Returns
        -------
        X : ndarray of shape (n_samples, n_features)
            2D array containing the `float`-type feature values. First dimension corresponds to the observation, while the
            second dimension corresponds to the feature.
        y : ndarray of shape (n_samples,)
            1D array containing the `int`-type label values. For a negative label equal to 0 and equal to 1
            for a positive label.
        """
        if random_state is not None:
            rng = np.random.default_rng(random_state)
        else:
            rng = self.rng

        # Generate the number of occurences for each class from a multinomial
        if self.random_imbalance:

            # To introduce additional variance to the class balance, use a beta prior for the binomial draws
            if self.imbalance_prior_a is not None and self.imbalance_prior_b is not None:
                p_prior = rng.beta(
                    self.imbalance_prior_a, self.imbalance_prior_b)
                p_weights = [p_prior, 1-p_prior]
            else:
                p_weights = self.weights

            instances = rng.multinomial(self.n_samples, p_weights)
            weights = instances/self.n_samples

        else:
            weights = self.weights

        n_useless = self.n_features - self.n_informative - \
            self.n_redundant - self.n_repeated
        n_clusters = self.n_classes * self.n_clusters_per_class

        # Distribute samples among clusters by weight
        n_samples_per_cluster = [
            int(self.n_samples *
                weights[k % self.n_classes] / self.n_clusters_per_class)
            for k in range(n_clusters)
        ]

        for i in range(self.n_samples - sum(n_samples_per_cluster)):
            n_samples_per_cluster[i % n_clusters] += 1

        # Initialize X and y
        X = np.zeros((self.n_samples, self.n_features))
        y = np.zeros(self.n_samples, dtype=int)

        # Centroids initialized as fixed points
        centroids = np.vstack(
            [np.zeros(self.n_informative), np.ones(self.n_informative)])

        # Set the euclidean distance between the centroids by scaling each feature equally
        if n_clusters == 2:
            eucl_d = np.linalg.norm(centroids[0]-centroids[1])
            centroids = centroids/eucl_d * self.class_d

        else:
            raise Exception("No setting yet for more than two clusters")

        # Initially draw informative features from the standard normal
        X[:, :self.n_informative] = rng.standard_normal(size=(
            self.n_samples, self.n_informative))

        # Create each cluster; a variant of make_blobs
        stop = 0
        for k, centroid in enumerate(centroids):
            start, stop = stop, stop + n_samples_per_cluster[k]
            y[start:stop] = k % self.n_classes  # assign labels
            # slice a view of the cluster
            X_k = X[start:stop, :self.n_informative]

            # The covariance between features is fixed as cov and the variance fixed as var
            A = np.full((self.n_informative, self.n_informative), self.cov)
            np.fill_diagonal(A, self.var)

            X_k[...] = np.dot(X_k, A)

            X_k += centroid  # shift the cluster to a centroid

        # Create redundant features
        if self.n_redundant > 0:
            B = 2 * rng.uniform(size=(self.n_informative,
                                      self.n_redundant)) - 1
            X[:, self.n_informative: self.n_informative + self.n_redundant] = np.dot(
                X[:, :self.n_informative], B
            )

        # Repeat some features
        if self.n_repeated > 0:
            n = self.n_informative + self.n_redundant
            indices = (
                (n - 1) * rng.uniform(size=self.n_repeated) + 0.5).astype(np.intp)
            X[:, n: n + self.n_repeated] = X[:, indices]

        # Fill useless features
        if n_useless > 0:
            X[:, -
                n_useless:] = rng.standard_normal((self.n_samples, n_useless))

        # Randomly flip labels with probability flip_y
        if self.flip_y >= 0.0:
            flip_mask = rng.uniform(size=self.n_samples) < self.flip_y
            y[flip_mask] = 1 - y[flip_mask]

        return X, y

    def create_train_test(self, random_state=None):
        """
        Creates a randomly sampled train and test set.

        Parameters
        -------
        random_state : RandomState instance
            ,default is None

        Returns
        -------
        dict[str, dict]
            returns a dictionary containing two dictionaries:
                * 'train' which contains the labels `y` and `X`
                * 'test' which contains the labels `y` and `X`
        """
        if random_state is not None:
            rng = np.random.default_rng(random_state)
        else:
            rng = self.rng

        X, y = self.make_classification(random_state=rng)
        split_seed = rng.integers(low=0, high=np.iinfo(np.uint32).max)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=self.train_size, random_state=split_seed)
        return {'train': {'X': X_train, 'y': y_train}, 'test': {'X': X_test, 'y': y_test}}

    def split_train_test(self, X, y, train_size, rng):
        train_indices = rng.choice(len(y), size=train_size, replace=False)
        train_mask = np.zeros(len(y), dtype=bool)
        train_mask[train_indices, ] = True

        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[~train_mask], y[~train_mask]
        return X_train, X_test, y_train, y_test

    def plot_blobs(self, X, y, contour=True, scatter=True):
        """
        Plots the coordinates corresponding to the first and second features. Optional contour or scatter settings.

        Parameters
        -------
        X : ndarray
            2D array containing the `float`-type feature values. First dimension corresponds to the observation, while the
            second dimension corresponds to the feature.
        y : ndarray
            1D array containing the `int`-type label values. For a negative label equal to 0 and equal to 1
            for a positive label.
        """
        if contour:
            data = pd.DataFrame(X)
            data['label'] = y
            sns.displot(data, x=0, y=1, hue='label', kind='kde')
        if scatter:
            plt.scatter(X[:, 0], X[:, 1], marker="o", c=y, s=25, edgecolor="k")
        plt.show()
