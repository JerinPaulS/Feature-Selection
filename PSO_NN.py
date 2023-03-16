import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyswarms as ps
from numpy import genfromtxt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn import metrics
import matplotlib.pyplot as plt
import WrapperFeatureSelection as WFS

class PSO_NN:

    n_inputs = None
    n_hidden = None
    n_classes = None
    data = None
    target = None
    num_samples = None
    X = None
    y = None

    def logits_function(self, p):
        global data, target, num_samples, n_inputs, n_classes, n_hidden, X, y
        """ Calculate roll-back the weights and biases
        Inputs
        ------
        p: np.ndarray
            The dimensions should include an unrolled version of the
            weights and biases.
        Returns
        -------
        numpy.ndarray of logits for layer 2
        """
        # Roll-back the weights and biases
        W1 = p[0:(n_inputs * n_hidden)].reshape((n_inputs,n_hidden))
        b1 = p[(n_inputs * n_hidden):(n_inputs * n_hidden) + 20].reshape((n_hidden,))
        W2 = p[(n_inputs * n_hidden) + 20:(n_inputs * n_hidden) + 60].reshape((n_hidden, n_classes))
        b2 = p[(n_inputs * n_hidden) + 60:222].reshape((n_classes,))

        # Perform forward propagation
        z1 = X.dot(W1) + b1  # Pre-activation in Layer 1
        a1 = np.tanh(z1)     # Activation in Layer 1
        logits = a1.dot(W2) + b2 # Pre-activation in Layer 2
        return logits          # Logits for Layer 2

    # Forward propagation
    def forward_prop(self, params):
        global data, target, num_samples, n_inputs, n_classes, n_hidden, X, y
        """Forward propagation as objective function
        This computes for the forward propagation of the neural network, as
        well as the loss.
        Inputs
        ------
        params: np.ndarray
            The dimensions should include an unrolled version of the
            weights and biases.
        Returns
        -------
        float
            The computed negative log-likelihood loss given the parameters
        """

        logits = self.logits_function(params)

        # Compute for the softmax of the logits
        exp_scores = np.exp(logits)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        # Compute for the negative log likelihood

        corect_logprobs = -np.log(probs[range(num_samples), self.y])
        loss = np.sum(corect_logprobs) / num_samples

        return loss

    def f(self, x):
        global data, target, num_samples, n_inputs, n_classes, n_hidden, X, y
        """Higher-level method to do forward_prop in the
        whole swarm.
        Inputs
        ------
        x: numpy.ndarray of shape (n_particles, dimensions)
            The swarm that will perform the search
        Returns
        -------
        numpy.ndarray of shape (n_particles, )
            The computed loss for each particle
        """
        n_particles = x.shape[0]
        j = [self.forward_prop(x[i]) for i in range(n_particles)]
        return np.array(j)

    def predict(self, X, p):
        global n_inputs, n_classes, n_hidden
        """
        Use the trained weights to perform class predictions.
        Inputs
        ------
        X: numpy.ndarray
            Input Iris dataset
        pos: numpy.ndarray
            Position matrix found by the swarm. Will be rolled
            into weights and biases.
        """

        # Roll-back the weights and biases
        W1 = p[0:(n_inputs * n_hidden)].reshape((n_inputs,n_hidden))
        b1 = p[(n_inputs * n_hidden):(n_inputs * n_hidden) + 20].reshape((n_hidden,))
        W2 = p[(n_inputs * n_hidden) + 20:(n_inputs * n_hidden) + 60].reshape((n_hidden,n_classes))
        b2 = p[(n_inputs * n_hidden) + 60:222].reshape((n_classes,))

        # Perform forward propagation
        z1 = X.dot(W1) + b1  # Pre-activation in Layer 1
        a1 = np.tanh(z1)     # Activation in Layer 1
        z2 = a1.dot(W2) + b2 # Pre-activation in Layer 2
        logits = z2          # Logits for Layer 2

        y_pred = np.argmax(logits, axis=1)
        return y_pred

    def main(self):
        global data, target, num_samples, n_inputs, n_classes, n_hidden, X, y
        data, target = WFS.read_data()
        # data, target = WFS.WrapperFeatureSelection()

        data = data.filter(['smoothness_mean', 'concavity_mean', 'symmetry_mean', 'compactness_se', 'symmetry_se', 'radius_worst', 'texture_worst'], axis=1)

        # print(data)
        # print(target)

        X = data.to_numpy()
        y = target.to_numpy()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

        n_inputs = X.shape[1]
        n_hidden = 20
        n_classes = 2

        print(n_inputs, y_test)

        num_samples = X.shape[0]

        # Initialize swarm
        options = {'c1': 0.5, 'c2': 0.3, 'w':0.9}

        # Call instance of PSO
        dimensions = (n_inputs * n_hidden) + (n_hidden * n_classes) + n_hidden + n_classes
        optimizer = ps.single.GlobalBestPSO(n_particles=150, dimensions=dimensions, options=options)

        # Perform optimization
        cost, pos = optimizer.optimize(self.f, iters=500)

        predicted_values = self.predict(X_test, pos)
        actual_values = y_test
        confusion_matrix = metrics.confusion_matrix(actual_values, predicted_values)
        cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])
        cm_display.plot()
        plt.show()
        print("Accuracy score",accuracy_score(actual_values, predicted_values))
        print("Precision score",precision_score(actual_values, predicted_values, pos_label=1))
        print("Recall score",recall_score(actual_values, predicted_values))
        print("F1 score",f1_score(actual_values, predicted_values))

obj = PSO_NN()
obj.main()