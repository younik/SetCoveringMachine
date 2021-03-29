# Set Covering Machine

The Set Covering Machine is a learning algorithm for binary classification. The classifier is a solution of the set cover problem minimizing the empirical risk.

This implementation follow the idea presented in the original paper: 
![Marchand, Mario & Shawe-Taylor, John & Brodley, E. & Danyluk, Andrea. (2003). The Set Covering Machine. Journal of Machine Learning Research. 3. 10.1162/jmlr.2003.3.4-5.723. ](https://www.jmlr.org/papers/volume3/marchand02a/marchand02a.pdf)

So the *sets* are *balls* generated from the training set and the set cover problem is solved using greedy algorithm (Chvátal, 1979), accordingly to the paper. Sets can be combined in conjunction or disjunction and this can be specified using the parameter *machine_type*.

## How to install

You can install the package using pip:

    pip install scmpy

## Example
    from sklearn.model_selection import train_test_split
    from sklearn.datasets import make_circles
    from scmpy import SCM

    # make some artificial data
    X, y = make_circles(n_samples=200, noise=0.2, factor=0.5, random_state=666)

    # split data in train and test
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=.3, random_state=7)

    scm = SCM(machine_type="disjunction")

    # fit the model
    scm.fit(X_train, y_train)

    # now the model is fitted, we can predict values

    # print accuracy
    print(scm.score(X_test, y_test))
