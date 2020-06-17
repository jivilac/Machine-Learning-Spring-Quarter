import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import sklearn
from sklearn.linear_model import LogisticRegression
import copy
from sklearn.ensemble import AdaBoostClassifier

#loading data
import numpy as np
np.random.seed(0)
mnist = sio.loadmat('mnist_data_new.mat')
train_data = mnist['train_data']
train_label = mnist['train_label'].reshape(-1)
test_data = mnist['test_data']
test_label = mnist['test_label'].reshape(-1)

#Q a)
#ADD CODE HERE: Reshaping the data

#ADD CODES ABOVE
#implement LR on reshaped data, C is the inverse of regularization strength
C = 1
clf = LogisticRegression(random_state=0, C = C, max_iter = 4000)
#For fair comparison with Adaboost in part c), we set the sample weight in LR to be 1/n
data_weight = np.ones((train_data.shape[0],)) / train_data.shape[0]
clf.fit(train_data, train_label, sample_weight = data_weight)
#ADD CODE HERE: Report the training and testing accuracy

#ADD CODES ABOVE

#Q b)
class Bagging(object):
def __init__(self, base_classifier, b_bootstrap, m_bootstrap_size, class_num = 10):
    self.base_classifier = base_classifier
    #b_bootstrap denotes b different bootstrap samples, same as B denoted in lecture notes
    self.b_bootstrap = b_bootstrap
    self.base_classifier_list = []
    self.class_num = class_num
    #set the size of each bootstrap sample as 4000, same as m denoted in lecture notes
    self.m_bootstrap_size = m_bootstrap_size

def fit(self , X_train , y_train ):
    for i in  range(self.b_bootstrap):
        # ADD CODES BELOW
        #bootstrap m samples from n training data with replacement
        
        # ADD CODES ABOVE
        #for fair comparison with a) and c), set the sample weight to be 1/m when fitting LR
        data_weight = np.ones((x_sub.shape[0],)) / x_sub.shape[0]
        clf = copy.deepcopy(self.base_classifier)
        #only for fair comparison with baseline, would not require for general bagging
        clf.fit(x_sub, y_sub, sample_weight = data_weight)
        self.base_classifier_list.append(clf)

def predict(self, X_test):
    #create a n x 10 matrix for storing the result
    pred = np.zeros((X_test.shape[0], self.class_num))
    pred_list = []
    for i in range(self.b_bootstrap):
        # ADD CODES BELOW
        
        # ADD CODES ABOVE
    return pred.argmax(axis = 1)

bagging = Bagging(LogisticRegression(random_state=0, C = C, max_iter = 4000), b_bootstrap = 5, m_bootstrap_size = 4000)
#ADD CODE HERE: Report the testing accuracy

#ADD CODES ABOVE

#Q c)
class AdaBoosting(object):
def __init__(self, base_classifier, b_bootstrap, class_num = 10):
    self.base_classifier = base_classifier
    #b_bootstrap denotes b different bootstrap samples, same as B in lecture notes
    self.b_bootstrap = b_bootstrap
    self.base_classifier_weight = []
    self.base_classifier_list = []
    self.class_num = class_num

def fit(self , X_train , y_train ):
    data_weight = np.ones((X_train.shape[0],)) / X_train.shape[0]
    for i in  range(self.b_bootstrap):
        clf = copy.deepcopy(self.base_classifier)
        clf.fit(X_train, y_train, sample_weight = data_weight)
        y_pred = clf.predict(X_train)
        #ADD CODES BELOW: updating the weight

        #ADD CODES ABOVE: updating the weight
        self.base_classifier_list.append(clf)
        self.base_classifier_weight.append(error_weight)

def predict(self, X_test):
    pred = np.zeros((X_test.shape[0], self.class_num))
    pred_list = []
    for i in range(self.b_bootstrap):
            #ADD CODES HERE
        pred[np.arange(y_pred.shape[0]),y_pred] += self.base_classifier_weight[i]
    return pred.argmax(axis = 1)

adaboost = AdaBoosting(LogisticRegression(random_state=0, C = C, max_iter = 4000), b_bootstrap = 10)
adaboost.fit(train_data, train_label)
#ADD CODES HERE: Report the testing accuracy

#ADD CODES ABOVE
