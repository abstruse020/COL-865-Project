import torch
import torch.nn as nn
from typing import Type
import numpy as np
from intervaltree import Interval, IntervalTree
from sklearn.base import BaseEstimator, OutlierMixin
from sklearn.utils.validation import check_array
from sklearn.datasets import make_blobs, make_multilabel_classification, load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import (
    linear_kernel,
    polynomial_kernel,
    rbf_kernel,
    sigmoid_kernel,
)
from sklearn.metrics import roc_auc_score, accuracy_score
from datetime import datetime
from time import time
import pandas as pd
import matplotlib.pyplot as plt

class FROCC(BaseEstimator, OutlierMixin):
    """FROCC classifier

        Parameters
        ----------
        num_clf_dim : int, optional
            number of random classification directions, by default 10
        epsilon : float, optional
            sepratation margin as a fraction of range, by default 0.1
        threshold : float, optional
            agreement threshold, by default 1
        kernel : callable, optional
            kernel function, by default dot
        precision : type, optional
            floating point precision to use, by default np.float16

        Examples
        ---------
        >>> import frocc, datasets
        >>> x, y, _, _ = datasets.gaussian()
        >>> clf = FROCC()
        >>> clf.fit(x)
        >>> preds = clf.predict(x)
        """

    def __init__(
        self,
        num_clf_dim: int = 10,
        epsilon: float = 0.1,
        threshold: float = 1,
        kernel: Type[np.dot] = lambda x, y: x.dot(y.T),
        precision: type = np.float32,
    ):
        self.num_clf_dim = num_clf_dim
        self.precision = precision
        self.epsilon = epsilon
        self.threshold = threshold
        self.kernel = kernel
        self.clf_dirs = None

    def get_intervals(self, projection):
        """Compute epsilon separated interval tree from projection

        Parameters
        ----------
        projection : 1-d array
            Projection array of points on a vector

        Returns
        -------
        IntervalTree
            epsilon separated interval tree
        """
        start = projection[0]
        end = projection[0]
        epsilon = (np.max(projection) - np.min(projection)) * self.epsilon
        tree = IntervalTree()
        for point in projection[1:]:
            if point < end + epsilon:
                end = point
            else:
                try:
                    end += 2 * np.finfo(self.precision).eps
                    tree.add(Interval(start, end))
                except ValueError:
                    # NULL interval
                    pass
                start = point
                end = point
        else:
            try:
                end += 2 * np.finfo(self.precision).eps
                tree.add(Interval(start, end))
            except ValueError:
                # NULL interval
                pass
        return tree

    def in_interval(self, tree, point):
        """Check membership of point in Interval tree

        Parameters
        ----------
        tree : IntervalTree
            Interval tree
        point : self.precision
            point to check membership

        Returns
        -------
        bool
            True if `point` lies within an Interval in IntervalTree
        """
        return tree.overlaps(point)

    def fit(self, x, y=None):
        """Train FROCC

        Parameters
        ----------
        x : ndarray
            Training points
        y : 1d-array, optional
            For compatibility, by default None

        Returns
        -------
        self
            Fitted classifier
        """
        x = check_array(x)
        self.feature_len = len(x[0])
        clf_dirs = np.random.standard_normal(size=(self.num_clf_dim, self.feature_len))
        norms = np.linalg.norm(clf_dirs, axis=1)
        self.clf_dirs = self.precision(clf_dirs / norms.reshape(-1, 1))
        projections = self.kernel(x, self.clf_dirs)  # shape should be NxD
        projections = np.sort(projections, axis=0)

        self.intervals = [
            self.get_intervals(projections[:, d]) for d in range(self.num_clf_dim)
        ]
        self.is_fitted_ = True
        return self

    def decision_function(self, x):
        """Returns agreement fraction for points in a test set

        Parameters
        ----------
        x : ndarray
            Test set

        Returns
        -------
        1d-array - float
            Agreement fraction of points in x
        """
        projections = self.kernel(x, self.clf_dirs)
        scores = []
        for v in projections:
            num_agree = len(
                [
                    clf_dim
                    for clf_dim in range(self.num_clf_dim)
                    if self.in_interval(self.intervals[clf_dim], v[clf_dim])
                ]
            )
            scores.append(num_agree / self.num_clf_dim)
        return np.array(scores)

    def predict(self, x):
        """Predictions of FROCC on test set x

        Parameters
        ----------
        x : ndarray
            Test set

        Returns
        -------
        1d-array - bool
            Prediction on Test set. False means outlier.
        """
        scores = self.decision_function(x)
        return scores >= self.threshold

    def fit_predict(self, x, y=None):
        """Perform fit on x and returns labels for x.

        Parameters
        ----------
        x : ndarray
            Input data.
        y : ignored, optional
            Not used, present for API consistency by convention.

        Returns
        -------
        1-d array - bool
            Predition on x. False means outlier.
        """
        return super().fit_predict(x, y=y)

    def size(self):
        """Returns storage size required for classifier

        Returns
        -------
        int
            Total size to store random vectors and intervals
        """
        clf_dir_size = self.clf_dirs.nbytes
        n_intervals = 0
        for itree in self.intervals:
            n_intervals += len(itree.all_intervals)

        if self.precision == np.float16:
            interval_size = n_intervals * 16 / 8
        if self.precision == np.float32:
            interval_size = n_intervals * 32 / 8
        if self.precision == np.float64:
            interval_size = n_intervals * 64 / 8
        if self.precision == np.float128:
            interval_size = n_intervals * 128 / 8

        return clf_dir_size + interval_size

    def __sizeof__(self):
        return self.size()

class FrcMLP(nn.Module):
    def __init__(self, input_features, hidden_features= 10, output_dimension = 1):
        super(FrcMLP, self).__init__()
        self.input_features = input_features
        self.hidden_features = hidden_features
        self.output_dimension = output_dimension
        self.l1 = nn.Linear(input_features, hidden_features)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_features, output_dimension)
        
    def forward(self, X):
        out = self.l1(X)
        out = self.relu(out)
        out = self.l2(out)
        return out
    
    def fit(self, X, y, num_epochs=100,lr = 0.01, X_test=None, y_test=None):
        # criterion = nn.CrossEntropyLoss()
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        prev_loss = 0
        for epoch in range(num_epochs):
            # X = X.to(device)
            # y = y.to(device)
            outputs = self(X)
            loss = criterion(outputs, y)
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if epoch %10:
                if X_test is not None:
                    with torch.no_grad():
                        test_outputs = self(X_test)
                        test_loss = criterion(test_outputs, y_test)
                    # print(f'test_loss:{test_loss}')
                if prev_loss !=0 and prev_loss-test_loss<0.001:
                    print('-'*10,'breaking frc mlp training','-'*10)
                    break
                prev_loss = test_loss
        return self
    
    # def predict(self, X):
    #     self.forward(X)
    #     _, pred = torch.max(outputs.data, 1)
    #     return pred

class FRMCC:
    def __init__(self, epsilon, dimension, kernel='sigmoid', outfile='frocc_output_file.txt', get_multiclass_prediction='max'):
        self.epsilon = epsilon
        self.dimension = dimension
        self.kernel = kernel
        self.outfile = outfile
        self.clf_dict = {}
        self.labels = []
        self.get_multiclass_prediction = get_multiclass_prediction
    
    def train_occ_frocc(self, my_args, x, y_occ, xtest=None, ytest_occ=None, run=1):
        if my_args['method'] == "frocc":
            clf = FROCC(
                num_clf_dim=my_args['dimension'], epsilon=my_args['epsilon'], kernel=my_args['kernel']
            )
            # x = x.toarray()
            # xtest = xtest.toarray()
        # elif args.method == "dfrocc":
        #     clf = dfrocc.DFROCC(
        #         num_clf_dim=args.dimension, epsilon=args.epsilon, kernel=kernel
        #     )
        # elif args.method == "sparse_dfrocc":
        #     clf = sparse_dfrocc.SDFROCC(
        #         num_clf_dim=args.dimension, epsilon=args.epsilon, kernel=kernel
        #     )
        # elif args.method == "pardfrocc":
        #     clf = pardfrocc.ParDFROCC(
        #         num_clf_dim=args.dimension, epsilon=args.epsilon, kernel=kernel
        #     )
        clf.fit(x[y_occ])
        scores = None
        if xtest is not None:
            scores = clf.decision_function(xtest)
            roc = roc_auc_score(ytest_occ, scores)
            result_dict = {
                    "Run ID": run,
                    "Dimension": my_args['dimension'],
                    "Epsilon": my_args['epsilon'],
                    "AUC of ROC": roc,
                }
            #pred = clf.predict(xtest)
            # print(f"{'ytest occ':10}|{'scores':8}|{'pred':8}")
            # for it in range(len(xtest))[:15]:
            #     print(f"{ytest_occ[it]:<10},{scores[it]:<8},{pred[it]:<8}")
            # print('ytest_occ:',ytest_occ.shape,', scores:', scores.shape)
            print(f'roc score: {roc}')
            # df.to_csv(my_args['outfile'])
        return clf, scores
        
    def fit(self, x, y, xtest = None, ytest = None, print_details = True):
        kernels = dict(
            zip(
                ["rbf", "linear", "sigmoid"],
                [rbf_kernel, linear_kernel, sigmoid_kernel],
        ))
        my_args = {
            'repetitions': 1,
            'dimension': self.dimension,
            'epsilon': self.epsilon,
            'kernel': kernels[self.kernel],
            'outfile': self.outfile,
            'method': 'frocc'
        }
        # kernel = kernels.get(my_args['kernel'])
        
        self.clf_dict = {}
        self.labels = pd.unique(y)
        # clf_scores = np.zeros((len(xtest), len(self.labels)))
        
        if print_details:
            print(f"x shape: {x.shape}, y shape: {y.shape}")
            print(f'Labels : {self.labels}, len: {len(self.labels)}')
            print(f"epsilon: {my_args['epsilon']}, kernel: {self.kernel},\nrepetitions: {my_args['repetitions']}, dimension: {my_args['dimension']}")
            # print('clf_scores shape', clf_scores.shape)
        
        for label in self.labels:
            # if print_details:
            #     print(f'For label: {label}')
            y_occ = y == label
            self.clf_dict[label], _ = self.train_occ_frocc(my_args, x, y_occ)
        
        if xtest is not None:
            y_hat = self.predict(xtest, ytest, print_acc = print_details)
            
        return self
    
    def predict(self, x, y = None, print_acc = True, path = None):
        clf_scores = np.zeros((len(x), len(self.labels)))
        for label_index, label in enumerate(self.labels):
            clf_scores[:,label_index] = self.clf_dict[label].decision_function(x)
            if y is not None:
                y_occ = y == label
                roc = roc_auc_score(y_occ, clf_scores[:,label_index])
                if print_acc:
                    print(f"label: {label}, roc: {roc}")
                if path is not None:
                    with open(path, '+a') as f:
                        f.write(f"label: {label}, roc: {roc}\n")
                        
        if self.get_multiclass_prediction == 'best_class_first':
            raise Exception(f"Not Implemented :{self.get_multiclass_prediction}")
            # y_hat = np.argmax(clf_scores, axis=1)
        elif self.get_multiclass_prediction == 'max':
            y_hat = np.argmax(clf_scores, axis=1)
        else:
            y_hat = np.argmax(clf_scores, axis=1)
        y_hat = self.labels[y_hat]
        # print('y_hat shape', y_hat.shape)
        if y is not None:
            acc = accuracy_score(y, y_hat)
            if print_acc:
                print(f"accuracy: {acc}")
            if path is not None:
                with open(path, '+a') as f:
                    f.write(f"accuracy: {acc}\n")
        return y_hat, clf_scores
    