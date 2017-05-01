#classifier.py

import numpy as np
from sklearn import svm

# Used to classify events from MFCC features
class SVMClassifier(object) :
    def __init__(self):
        super(SVMClassifier, self).__init__()

        self.clf = svm.SVC()

        # Map integer SVM classes to event labels
        self.class2label = {
            -1: "",
            0: "kick",
            1: "hihat",
            2: "snare"
        }

    # Trains SVM using given matrix of MFCC features and labels
    def train(self, mfccs, labels):
        print "Training SVM on %d samples..." % len(labels)
        self.clf.fit(mfccs, labels)
        print "Completed SVM training"

    # Classify event given vector of MFCCs for a given frame
    def classify(self, mfccs):
        return self.class2label[self.clf.predict([mfccs])[0]]

