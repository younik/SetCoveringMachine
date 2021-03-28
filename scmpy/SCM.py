import numpy as np


class SCM:
    """
    SCM(max_features=4, machine_type="conjunction", penalty_value=2.0)

    This class implement the Set Covering Machine classifier with data-dependent balls.

    For further details about this classifier: https://www.jmlr.org/papers/volume3/marchand02a/marchand02a.pdf

    Parameters
    ----------
    max_features : int, default=4
        Max features to combine to build the model.
        In this context, the term feature has to be intended as defined in the paper.

    machine_type : {'conjunction', 'disjunction'}, default='conjunction'
        Logical connective to combine the different features.

    penalty_value : float, default=2.0
        The penalty value for positive class misclassification. It is used to compute the utility score of features.

    Attributes
    ----------
    max_features : int
        max_features value passed as argument.

    machine_type : int
        Assume 0 if the machine type is 'conjunction', 1 otherwise. Also identify the positive class.

    penalty_value : float
        penalty_value value passed as argument.

    model : list of Feature
        List of Feature objects forming the model.

    """

    def __init__(self, max_features=4, machine_type="conjunction", penalty_value=2.0):
        if machine_type != "conjunction" and machine_type != "disjunction":
            raise Exception("machine_type must be either conjunction or disjunction")

        self.max_features = max_features
        self.machine_type = machine_type == "conjunction"
        self.penalty_value = penalty_value
        self.model = []

    def _init_features(self, x, y, epsilon=2 ** -28):
        features = []
        for center, label in zip(x, y):
            for border_point in x:
                radius = np.linalg.norm(center - border_point)
                if radius > epsilon:
                    radius = radius + epsilon if label == self.machine_type else radius - epsilon
                    features.append(Feature(center, radius, label))

        return features

    def fit(self, x, y):
        """
        fit(self, x, y)

        This method compute the model for the given data.

        Parameters
        ----------
        x : array-like of shape (n_queries, n_dim)
            The data of the training set.

        y : array-like of shape (n_queries,)
            The classes associated to x.

        """

        if len(x) != len(y):
            raise Exception("Points and labels must have the same length")

        if np.any(np.logical_and(y != 1, y != 0)):
            raise Exception("Labels must be either 0 or 1")

        self.model = []
        positive_label = self.machine_type
        negative_label = 1 - positive_label
        positive_points = x[np.nonzero(y == positive_label)]
        negative_points = x[np.nonzero(y == negative_label)]

        features = self._init_features(x, y)

        while len(negative_points) != 0 and len(self.model) < self.max_features:
            best_feature = max(features, key=lambda f: f.utility(positive_points, negative_points, self.penalty_value,
                                                                 self.machine_type))

            self.model.append(best_feature)
            negative_points = negative_points[np.logical_not(best_feature(negative_points))]
            positive_points = positive_points[np.logical_not(best_feature(positive_points))]

    def predict(self, x):
        """
        predict(self, x)

        Predict the class label for each target in x.

        Parameters
        ----------
        x : array-like of shape (n_queries, n_dim)
            Target samples.

        Returns
        -------
        y : array of shape (n_queries,)
            Predicted class labels for each given target sample.

        Examples
        --------
        >>> from scmpy import SCM
        >>> from sklearn.model_selection import train_test_split
        >>> from sklearn.datasets import make_circles
        >>> data, labels = make_circles(n_samples=200, noise=0.2, factor=0.5, random_state=666)
        >>> X_train, X_test, y_train, y_test = \
                 train_test_split(data, labels, test_size=.3, random_state=7)
        >>> scm = SCM(max_features=4, machine_type="disjunction")
        >>> scm.fit(X_train, y_train)
        >>> predicted_classes = scm.predict(X_test)

        """

        ys = [cls(x) for cls in self.model]
        if self.machine_type:
            return np.all(ys, axis=0)
        else:
            return np.any(ys, axis=0)

    def predict_proba(self, x):
        """
        predict_proba(self, x)

        Estimate the probabilities of belonging to the classes for each target in x.

        Parameters
        ----------
        x : array-like of shape (n_queries, n_dim)
            Target samples.

        Returns
        -------
        y : array of shape (n_queries, 2)
            Estimated probabilities of belonging to class 0 and class 1 respectively for each given target sample.

        Notes
        -----
        Set Covering Machine does not compute probabilities.
        This method assign probability of 1 to the predicted class and 0 the the other.

        Examples
        --------
        >>> from scmpy import SCM
        >>> from sklearn.model_selection import train_test_split
        >>> from sklearn.datasets import make_circles
        >>> data, labels = make_circles(n_samples=200, noise=0.2, factor=0.5, random_state=666)
        >>> X_train, X_test, y_train, y_test = \
                 train_test_split(data, labels, test_size=.3, random_state=7)
        >>> scm = SCM(max_features=4, machine_type="disjunction")
        >>> scm.fit(X_train, y_train)
        >>> estimated_probabilities = scm.predict_proba(X_test)
        >>> estimated_probabilities.shape
        (60, 2)

        """

        y = self.predict(x).reshape(1, -1).T
        return np.append(1 - y, y, axis=1)

    def score(self, x, y):
        """
        score(self, x, y)

        Return the accuracy score of the given test data and labels.

        Parameters
        ----------
        x : array-like of shape (n_queries, n_dim)
            Test samples.

        y : array-like of shape (n_queries,)
            True class of x.

        Returns
        -------
        score : float
            The accuracy of the predicted labels of x with respect to y.

        Examples
        --------
        >>> from scmpy import SCM
        >>> from sklearn.model_selection import train_test_split
        >>> from sklearn.datasets import make_circles
        >>> data, labels = make_circles(n_samples=200, noise=0.2, factor=0.5, random_state=666)
        >>> X_train, X_test, y_train, y_test = \
                 train_test_split(data, labels, test_size=.3, random_state=7)
        >>> scm = SCM(max_features=4, machine_type="disjunction")
        >>> scm.fit(X_train, y_train)
        >>> accuracy = scm.score(X_test, y_test)
        >>> accuracy
        0.8666666666666667

        """

        return np.count_nonzero(self.predict(x) == y) / len(y)


class Feature:
    """
    Feature(center, radius, label)

    An instance of this class represent a 'data-dependent ball'.

    Please, refer to paper for a formal definition of feature.

    """

    def __init__(self, center, radius, label):
        self.radius = radius
        self.center = center
        self.label = label

    def __call__(self, point):
        if self.label == 0:
            return np.linalg.norm(self.center - point, axis=1) > self.radius
        else:
            return np.linalg.norm(self.center - point, axis=1) <= self.radius

    def utility(self, positive_points, negative_points, penalty_value, positive_label):
        negative_covered = np.count_nonzero(self(negative_points) == 1 - positive_label)
        positive_wrong = np.count_nonzero(self(positive_points) != positive_label)
        return negative_covered - penalty_value * positive_wrong
