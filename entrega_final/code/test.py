import clasifiers as clf
import features as feat
import carga as carga
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score, roc_curve, auc

import time
import joblib
from tabulate import tabulate


def test_classifier(classifier: clf.Classifier, X_train, X_test, y_train, y_test):
    train_st = time.time()
    time_id = int(time.time())
    classifier.train(X_train, y_train)
    joblib.dump(classifier, f'./model/{classifier.__class__.__name__}{time_id}.pkl')
    train_et = time.time()
    y_pred = classifier.classify(X_test)
    report_test = classification_report(y_test, y_pred, output_dict=True)

    return y_test, y_pred, report_test, train_et - train_st

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

    X_train = X_train[10:100]
    y_train = y_train[10:100]

    X_test = X_test[10:30]
    y_test = y_test[10:30]

    features_methods = [feat.reshape() , feat.HOGPrepocess() ,  feat.HAARPreprocess()]
    #podemos pasar parametros si es necesario

    knn_classifier = clf.KNNClassifier()
    dtree_classifier = clf.DTreeClassifier()
    logistic_regression_classifier = clf.LogisticRegressionClassifier()
    rf_classifier = clf.RFClassifier()
    boosting_classifier = clf.BoostingClassifier()

    classifiers = [knn_classifier,  logistic_regression_classifier]#,dtree_classifier, rf_classifier, boosting_classifier]

    results = []
    for fea in features_methods:
        X_train_prep = fea.preprocess_imgs(X_train)
        X_test_prep = fea.preprocess_imgs(X_test)
        for cls in classifiers:
            y_test, y_pred, report_test, ti = test_classifier(cls, X_train_prep, X_test_prep, y_train, y_test)
            conf_matrix = confusion_matrix(y_test, y_pred)
            fpr, tpr, thresholds = roc_curve(y_test, y_pred)
            roc_auc = auc(fpr, tpr)
            tp, fp, fn, tn = conf_matrix.ravel()
            #print(f"Testing {cls.__class__.__name__} con {prep.__class__.__name__} {report_test['accuracy']} {report_test['macro avg']['precision']} {report_test['macro avg']['recall']} {report_test['macro avg']['f1-score']} {ti}")
            results.append([
                cls.__class__.__name__,
                fea.__class__.__name__,
                report_test['accuracy'],
                report_test['1']['precision'],
                tp / (tp + fn),
                fp / (tn + fp),
                roc_auc,
                balanced_accuracy_score(y_test,y_pred),
                report_test['1']['f1-score'],
                ti
            ])  
            #print(f"Precision: {report_test}")
    headers = ["Classifier", "Preprocessing", "Accuracy", "Precision", "Recall/TPR", "FPR", "F1-Score", "ROC curve (area)", "Balanced Accuracy", "Time Train"]
    print(tabulate(results, headers, tablefmt="grid"))

  