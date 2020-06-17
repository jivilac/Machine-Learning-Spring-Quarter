#python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_circles

# create dataset
X, y = make_circles(n_samples=400, noise=0.2, factor=0.5, random_state=0)
X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4, random_state=42)



from sklearn.tree import DecisionTreeClassifier
# ADD CODES HERE:
#   clf = ...

# plotting
x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
h = .02
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
Z = Z.reshape(xx.shape)
plt.figure()
plt.title("Stump", fontsize='small')
cm = plt.cm.RdBu
cm_bright = ListedColormap(['#FF0000', '#0000FF'])
plt.contourf(xx, yy, Z, cmap=cm, alpha=.8)
plt.contour(xx, yy, np.round(Z), 0)
plt.scatter(X_train[:, 0], X_train[:, 1], marker='o', c=y_train, cmap=cm_bright, edgecolors='k')
plt.scatter(X_test[:, 0], X_test[:, 1], marker='o', c=y_test, cmap=cm_bright, alpha=0.6, edgecolors='k')



import copy

class Bagging():
    def __init__(self, base_classifier, n_bootstrap, portion):
        self.base_classifier = base_classifier
        self.n_bootstrap = n_bootstrap
        self.portion = portion
        self.base_classifier_list = []

    def fit(self, X_train, y_train):
        for i in range(self.n_bootstrap):
            clf = copy.deepcopy(self.base_classifier)
            # ADD CODES HERE
            #
            self.base_classifier_list.append(clf)

    def predict_proba(self, X_test):

    # RETURN AVERAGED PREDICTED PROBABILITY FOR CLASS 1
    #   (THE SECOND CASE ON SLIDE 4 OF LECTURE 11)

    def score(self, X_test, y_test):
        y_pred = self.predict_proba(X_test)
        acc = sum(np.round(y_pred) == y_test) / len(y_test)
        return acc


clf = Bagging(DecisionTreeClassifier(max_depth=1),n_bootstrap=200, portion=0.8)
clf.fit(X_train, y_train)
# PLOT FIGURE HERE (USE SIMILAR CODES AS ABOVE)
#




class Boosting():
    def __init__(self, base_classifier, n_iterations):
        self.base_classifier = base_classifier
        self.n_iterations = n_iterations
        # ADD CODES HERE
        #

    def fit(self, X_train, y_train):
        y_train = 2 * y_train - 1
        # ADD CODES HERE
        #

    def predict_label(self, X_test):

    # RETURN A NUMBER BETWEEN -1 AND 1
    #

    def score(self, X_test, y_test):
        y_test = 2 * y_test - 1
        y_pred = self.predict_label(X_test)
        acc = sum(np.sign(y_pred) == y_test) / len(y_test)
        return acc


clf = Boosting(DecisionTreeClassifier(max_depth=1), n_iterations=200)
clf.fit(X_train, y_train)
# PLOT FIGURE HERE
#