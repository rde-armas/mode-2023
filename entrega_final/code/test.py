import clasifiers as clf
import preprocessing as pre
import carga as carga
import numpy as np
from sklearn.model_selection import train_test_split
#from sklearn.metrics import confusion_matrix, precision_score, recall_score, roc_curve
import time
#from tqdm.auto import tqdm
import joblib


def test_classifier(classifier: clf.Classifier, X_train, X_test, y_train, y_test):
    
    train_st = time.time()
    classifier.train(X_train, y_train)
    joblib.dump(classifier, f'./model/{classifier.__class__.__name__}.pkl')
    train_et = time.time()

    Y_pred = classifier.classify(X_test)

    return y_test, Y_pred, train_et - train_st

if __name__ == "__main__":
    positive_faces = carga.positive_patches()
    negative_faces = carga.negative_patches('./imagenes/')

    # Etiquetas para caras (clase positiva)
    positive_labels = np.tile(1, 13233)

    # Etiquetas para fondos (clase negativa)
    negative_labels = np.tile(0, 13120)

    samples = np.concatenate((positive_faces, negative_faces))
    labels = np.concatenate((positive_labels, negative_labels))
    X_train, X_test, y_train, y_test = train_test_split(samples, labels, test_size=0.20, random_state=42)


    preprocessing_methods = [pre.HOGPrepocess(), pre.HAARPreprocess(), ]
    #podemos pasar parametros si es necesario

    knn_classifier = clf.KNNClassifier()
    dtree_classifier = clf.DTreeClassifier()
    logistic_regression_classifier = clf.LogisticRegressionClassifier()
    rf_classifier = clf.RFClassifier()
    boosting_classifier = clf.BoostingClassifier()

    classifiers = [knn_classifier, dtree_classifier, logistic_regression_classifier, rf_classifier, boosting_classifier]

    X_train_reshape = np.reshape(X_train,(21082, 2914))
    X_test_resahape = np.reshape(X_test, (5271, 2914))
    results_logistic_regression = test_classifier(logistic_regression_classifier, X_train_reshape, X_test_resahape, y_train, y_test ) 
  