import code.clasifiers as clf
import code.preprocessing as pre
from sklearn.metrics import confusion_matrix, precision_score, recall_score, roc_curve
import time

def test_classifier(classifier: clf.Classifier, X_train, Y_train, X_test, Y_test):
    
    train_st = time.time()
    classifier.train(X_train, Y_train)
    train_et = time.time()

    Y_pred = classifier.classify(X_test)

    return Y_test, Y_pred, train_et - train_st

if __name__ == "__main__":
    preprocessing_methods = [pre.HOGPreprocess(), pre.HAARPreprocess()]

    #podemos pasar parametros

    classifier_knn = get_clas