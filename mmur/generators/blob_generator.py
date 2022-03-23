"""
Generate samples of synthetic data sets.
"""
#%%
# Authors: B. Thirion, G. Varoquaux, A. Gramfort, V. Michel, O. Grisel,
#          G. Louppe, J. Nothman
# License: BSD 3 clause

import numbers
import array
from collections.abc import Iterable
import random

import numpy as np
from pyrsistent import v
from scipy import linalg
import scipy.sparse as sp

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.utils import check_array, check_random_state
from sklearn.utils import shuffle as util_shuffle
from sklearn.utils.random import sample_without_replacement



def _generate_hypercube(samples, dimensions, rng):
    """Returns distinct binary samples of length dimensions."""
    if dimensions > 30:
        return np.hstack(
            [
                rng.randint(2, size=(samples, dimensions - 30)),
                _generate_hypercube(samples, 30, rng),
            ]
        )
    out = sample_without_replacement(2 ** dimensions, samples, random_state=rng).astype(
        dtype=">u4", copy=False
    )
    out = np.unpackbits(out.view(">u1")).reshape((-1, 32))[:, -dimensions:]
    return out

class BlobGenerator:
    """Generate a random n-class classification problem.

        This initially creates clusters of points normally distributed (std=1)
        about vertices of an ``n_informative``-dimensional hypercube with sides of
        length ``2*class_sep`` and assigns an equal number of clusters to each
        class. It introduces interdependence between these features and adds
        various types of further noise to the data.

        Without shuffling, ``X`` horizontally stacks features in the following
        order: the primary ``n_informative`` features, followed by ``n_redundant``
        linear combinations of the informative features, followed by ``n_repeated``
        duplicates, drawn randomly with replacement from the informative and
        redundant features. The remaining features are filled with random noise.
        Thus, without shuffling, all useful features are contained in the columns
        ``X[:, :n_informative + n_redundant + n_repeated]``.

        Read more in the :ref:`User Guide <sample_generators>`.

        Parameters
        ----------
        n_samples : int, default=100
            The number of samples.

        n_features : int, default=20
            The total number of features. These comprise ``n_informative``
            informative features, ``n_redundant`` redundant features,
            ``n_repeated`` duplicated features and
            ``n_features-n_informative-n_redundant-n_repeated`` useless features
            drawn at random.

        n_informative : int, default=2
            The number of informative features. Each class is composed of a number
            of gaussian clusters each located around the vertices of a hypercube
            in a subspace of dimension ``n_informative``. For each cluster,
            informative features are drawn independently from  N(0, 1) and then
            randomly linearly combined within each cluster in order to add
            covariance. The clusters are then placed on the vertices of the
            hypercube.

        n_redundant : int, default=2
            The number of redundant features. These features are generated as
            random linear combinations of the informative features.

        n_repeated : int, default=0
            The number of duplicated features, drawn randomly from the informative
            and the redundant features.

        n_classes : int, default=2
            The number of classes (or labels) of the classification problem.

        n_clusters_per_class : int, default=2
            The number of clusters per class.

        weights : array-like of shape (n_classes,) or (n_classes - 1,),\
                default=None
            The proportions of samples assigned to each class. If None, then
            classes are balanced. Note that if ``len(weights) == n_classes - 1``,
            then the last class weight is automatically inferred.
            More than ``n_samples`` samples may be returned if the sum of
            ``weights`` exceeds 1. Note that the actual class proportions will
            not exactly match ``weights`` when ``flip_y`` isn't 0.

        class_sep : float, default=1.0
            The factor multiplying the hypercube size.  Larger values spread
            out the clusters/classes and make the classification task easier.

        hypercube : bool, default=True
            If True, the clusters are put on the vertices of a hypercube. If
            False, the clusters are put on the vertices of a random polytope.

        shift : float, ndarray of shape (n_features,) or None, default=0.0
            Shift features by the specified value. If None, then features
            are shifted by a random value drawn in [-class_sep, class_sep].

        scale : float, ndarray of shape (n_features,) or None, default=1.0
            Multiply features by the specified value. If None, then features
            are scaled by a random value drawn in [1, 100]. Note that scaling
            happens after shifting.

        shuffle : bool, default=True
            Shuffle the samples and the features.

        random_state : int, RandomState instance or None, default=None
            Determines random number generation for dataset creation. Pass an int
            for reproducible output across multiple function calls.
            See :term:`Glossary <random_state>`.

        var : float, 
            specifies the variance of each feature

        cov : 
            float, specifies the covariance between each pair of features

        flip_y : float, default=0.01
            The fraction of samples whose class is assigned randomly. Larger
            values introduce noise in the labels and make the classification
            task harder. Note that the default setting flip_y > 0 might lead
            to less than ``n_classes`` in y in some cases.

        random_imbalance : bool, default = False
            If set to True, the data sample class balance is stochastic, with the 
            number of draws for each class Multinomially distributed with 
            probabilities set to ``weights``.


        Returns
        -------
        X : ndarray of shape (n_samples, n_features)
            The generated samples.

        y : ndarray of shape (n_samples,)
            The integer labels for class membership of each sample.

        """

    def __init__(
        self,
        n_samples=100,
        n_features=20,
        n_informative=2,
        n_redundant=2,
        n_repeated=0,
        n_classes=2,
        n_clusters_per_class=2,
        weights=None,
        flip_y=0.01,
        class_sep=1.0,
        hypercube=True,
        shift=0.0,
        scale=1.0,
        shuffle=True,
        random_state=None,
        var = 1,
        cov = 0,
        random_imbalance = False):

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

        self.n_samples = n_samples
        self.n_features = n_features
        self.n_informative = n_informative
        self.n_redundant = n_redundant
        self.n_repeated = n_repeated
        self.n_classes = n_classes
        self.n_clusters_per_class = n_clusters_per_class
        self.flip_y = flip_y
        self.class_sep = class_sep
        self.hypercube = hypercube
        self.shift = shift
        self.scale = scale
        self.shuffle = shuffle
        self.var = var
        self.cov = cov
        self.random_imbalance = random_imbalance
        self.generator = check_random_state(random_state)

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


    def make_classification(self):

        if self.random_imbalance:
            instances = self.generator.multinomial(self.n_samples,self.weights)
            weights = instances/self.n_samples
        else:
            weights = self.weights

        n_useless = self.n_features - self.n_informative - self.n_redundant - self.n_repeated
        n_clusters = self.n_classes * self.n_clusters_per_class

        # Distribute samples among clusters by weight
        n_samples_per_cluster = [
            int(self.n_samples * weights[k % self.n_classes] / self.n_clusters_per_class)
            for k in range(n_clusters)
        ]

        for i in range(self.n_samples - sum(n_samples_per_cluster)):
            n_samples_per_cluster[i % n_clusters] += 1

        # Initialize X and y
        X = np.zeros((self.n_samples, self.n_features))
        y = np.zeros(self.n_samples, dtype=int)

        # Build the polytope whose vertices become cluster centroids
        # centroids = _generate_hypercube(n_clusters, n_informative, generator).astype(
        #     float, copy=False
        # )

        #change centroids to fixed points
        centroids = np.vstack([np.zeros(self.n_informative),np.ones(self.n_informative)])

        centroids *= 2 * self.class_sep
        centroids -= self.class_sep
        if not self.hypercube:
            centroids *= self.generator.rand(n_clusters, 1)
            centroids *= self.generator.rand(1, self.n_informative)

        # Initially draw informative features from the standard normal
        X[:, :self.n_informative] = self.generator.randn(self.n_samples, self.n_informative)

        # Create each cluster; a variant of make_blobs
        stop = 0
        for k, centroid in enumerate(centroids):
            start, stop = stop, stop + n_samples_per_cluster[k]
            y[start:stop] = k % self.n_classes  # assign labels
            X_k = X[start:stop, :self.n_informative]  # slice a view of the cluster

            #Original covariance matrix is randomized
            # A = 2 * generator.rand(n_informative, n_informative) - 1
            # A = np.identity(n_informative) #uncorrelated features

            #Here I change the covariance matrix such that the covariance between features is fixex as cov
            #and the variance fixed as var

            A=np.full((self.n_informative, self.n_informative), self.cov)
            np.fill_diagonal(A,self.var)

            X_k[...] = np.dot(X_k, A)  # introduce covariance

            X_k += centroid  # shift the cluster to a vertex

        # Create redundant features
        if self.n_redundant > 0:
            B = 2 * self.generator.rand(self.n_informative, self.n_redundant) - 1
            X[:, self.n_informative : self.n_informative + self.n_redundant] = np.dot(
                X[:, :self.n_informative], B
            )

        # Repeat some features
        if self.n_repeated > 0:
            n = self.n_informative + self.n_redundant
            indices = ((n - 1) * self.generator.rand(self.n_repeated) + 0.5).astype(np.intp)
            X[:, n : n + self.n_repeated] = X[:, indices]

        # Fill useless features
        if n_useless > 0:
            X[:, -n_useless:] = self.generator.randn(self.n_samples, n_useless)

        # Randomly replace labels
        if self.flip_y >= 0.0:
            flip_mask = self.generator.rand(self.n_samples) < self.flip_y
            y[flip_mask] = self.generator.randint(self.n_classes, size=flip_mask.sum())

        # Randomly shift and scale
        if self.shift is None:
            self.shift = (2 * self.generator.rand(self.n_features) - 1) * self.class_sep
        X += self.shift

        if self.scale is None:
            self.scale = 1 + 100 * self.generator.rand(self.n_features)
        X *= self.scale

        if self.shuffle:
            # Randomly permute samples
            X, y = util_shuffle(X, y, random_state=self.generator)

            # Randomly permute features
            indices = np.arange(self.n_features)
            self.generator.shuffle(indices)
            X[:, :] = X[:, indices]

        return X, y

#%%

if __name__ == '__main__':
    generator = BlobGenerator(
        n_samples=1000,
        n_features=2,
        n_informative=2,
        n_redundant=0,
        n_repeated=0,
        n_classes=2,
        n_clusters_per_class=1,
        weights=[0.5,0.5],
        flip_y=0.01,
        class_sep=1.0,
        scale=1.0,
        shuffle=True,
        random_state=123,
        var = 1,
        cov = 0,
        random_imbalance=True)

    X1,Y1 = generator.make_classification()

    import matplotlib.pyplot as plt
    plt.scatter(X1[:, 0], X1[:, 1], marker="o", c=Y1, s=25, edgecolor="k", alpha=0.5)
