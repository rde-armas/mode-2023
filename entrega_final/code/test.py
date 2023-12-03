import clasifiers as clf
import features as feat
import carga as carga
import numpy as np
import random
import os
import itertools
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score, roc_curve, auc

import pandas as pd

import time
import joblib
from tqdm.auto import tqdm

# import warnings

# # Ignorar todas las advertencias (puede no ser recomendable en todos los casos)
# warnings.filterwarnings("ignore")


def get_classifiers(parameters_dict: dict, Classifier):
    if len(parameters_dict) == 0:
        return []
    parameters = list(map(lambda x: list(zip(parameters_dict.keys(), x)), itertools.product(*parameters_dict.values())))
    classifiers = [Classifier(x) for x in parameters]
    return classifiers

def test_single_classifier(classifier: clf.Classifier, fea: feat.Features, X_train, X_test, y_train, y_test):
     # va a faltar agregar el nombre de los parametros de los modelos
    train_st = time.time()
    classifier.train(X_train, y_train)
    joblib.dump(classifier, f'./model/{classifier.__class__.__name__}{classifier.get_metaparameters()}{fea.__class__.__name__}.pkl')
    train_et = time.time()
    y_pred = classifier.classify(X_test)
    report_test = classification_report(y_test, y_pred, output_dict=True)

    return y_test, y_pred, report_test, train_et - train_st


# Guarda los resultados de los experimentos
def save_result(cls: clf.Classifier , fea: feat.Features, X_test, y_test, report_test, ti):
    headers = ["Classifier", "Preprocessing", "Accuracy", "Precision", "Recall/TPR", "FNR", "TNR", "FNR", "F1-Score", "ROC curve (area)", "Balanced Accuracy", "Time Train"]
    
    y_pred = cls.classify(X_test)
    conf_matrix = confusion_matrix(y_test, y_pred)
    tp, fp, fn, tn = conf_matrix.ravel()
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)

    roc_auc = auc(fpr, tpr)
   
    # Crear un diccionario con los datos
    name = cls.__class__.__name__ + cls.get_metaparameters()
    data = {
        "Classifier": [name],
        "Preprocessing": [fea.__class__.__name__],
        "Accuracy": [report_test['accuracy']],
        "Precision": [report_test['1']['precision']],
        "Recall/TPR": [tp / (tp + fn)],
        "FPR": [fp / (tn + fp)],
        "FNR": [tp / (tp +fn)],
        "TNR": [tp / (tp + fn)],    
        "F1-Score": [report_test['1']['f1-score']],
        "ROC curve (area)": [roc_auc],
        "Balanced Accuracy": [balanced_accuracy_score(y_test,y_pred)],
        "Time Train": [ti]
    }
    
    df = pd.DataFrame(columns=headers, data=data)

    # Save the updated DataFrame to the CSV file
    df.to_csv("./results/results.csv", index=False, header=not os.path.exists("./results/results.csv"), mode='a')


def test_classifiers(pro_train: int, prop_test: int, test_size_positive = 0.1):

    # Cargar rostros
    positive_faces = carga.positive_patches()
    # Mezclar los rostros
    random.shuffle(positive_faces)

    # Calcular los índices para la división del 90% y 10%
    index_90 = int(len(positive_faces) * (1 - test_size_positive))
    
    # Dividir la lista en dos sublistas
    positive_faces_train = positive_faces[:index_90]
    positive_faces_test = positive_faces[index_90:]
    
    # Calcula la cantidad de fondos para train y test
    train_amount = int(positive_faces.shape[0] * (1 - test_size_positive) * pro_train) 
    test_amount = int(positive_faces.shape[0] * test_size_positive * prop_test)
    # Cargar fondos
    negative_faces_train = carga.negative_patches(train_amount)
    negative_faces_test = carga.negative_patches(test_amount)

    # Etiquetas para caras (clase positiva)
    positive_labels_train = np.tile(1, positive_faces_train.shape[0])
    positive_labels_test = np.tile(1, positive_faces_test.shape[0])

    # Etiquetas para fondos (clase negativa)
    negative_labels_train = np.tile(0, negative_faces_train.shape[0])
    negative_labels_test = np.tile(0, negative_faces_test.shape[0])

    samples_train = np.concatenate((positive_faces_train, negative_faces_train))
    labels_train = np.concatenate((positive_labels_train, negative_labels_train))

    samples_test = np.concatenate((positive_faces_test, negative_faces_test))
    labels_test = np.concatenate((positive_labels_test, negative_labels_test))

    X_train, _ , y_train, _ = train_test_split(samples_train, labels_train, test_size = 0.001)
    _, X_test, _, y_test = train_test_split(samples_test, labels_test, test_size = 0.999)
    
    knn_parameters = {'n_neighbors':[5, 3, 7, 10], 'weights':['uniform', 'distance'], 'p':[1, 2]}
    dtree_parameters = {'criterion':['gini', 'entropy', 'log_loss'], 'max_depth': [3, 5, 7], 'min_samples_split': [2, 3, 5], 'min_samples_leaf': [1, 2, 3], 'max_features': ['sqrt', 'log2']}
    logistic_parameters = {'penalty': ['l2'], 'solver': ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga'], 'max_iter': [100, 150, 200], 'multi_class': ['auto', 'ovr'] }
    logistic_parameters_2 = {'penalty': ['l1'], 'solver': ['liblinear', 'saga'], 'max_iter': [100, 150, 200], 'multi_class': ['auto', 'ovr'] }
    logistic_parameters_3 = {'penalty': ['elasticnet'], 'solver': ['newton-cg', 'saga'], 'max_iter': [100, 150, 200], 'multi_class': ['auto', 'ovr', 'multinomial'] }
    rf_parameters = {'n_estimators': [100, 150, 200], 'criterion': ['gini', 'entropy', 'log_loss'], 'max_depth': [2, 5, 7], 'min_samples_split': [2, 3, 5], 'max_features': ['sqrt', 'log2', None]}
    #rf_parameters_2 = {'n_estimators': [100, 150, 200], 'criterion': ['gini', 'entropy', 'log_loss'], 'max_depth': [2, 5, 7], 'min_samples_split': [2, 3, 5], 'max_features': ['sqrt', 'log2', None]}

    knn_classifiers = get_classifiers(knn_parameters, clf.KNNClassifier )
    dtree_classifiers = get_classifiers(dtree_parameters, clf.DTreeClassifier)
    logistic_classifiers = get_classifiers(logistic_parameters, clf.LogisticRegressionClassifier)
    logistic_classifiers_2 = get_classifiers(logistic_parameters_2, clf.LogisticRegressionClassifier)
    logistic_classifiers_3 = get_classifiers(logistic_parameters_3, clf.LogisticRegressionClassifier)
    rf_classifiers = get_classifiers(rf_parameters, clf.RFClassifier)

    classifiers =  knn_classifiers
    features_methods = [feat.HAARPreprocess()] #, feat.HAARPreprocess()]

    with tqdm(total=len(features_methods), position=0, leave=False) as pbar1:
        for fea in features_methods:
            pbar1.set_description(f"Preprocesser: {fea.__class__.__name__}")
            pbar1.set_postfix({'state': 'Preprocessing'}, refresh=True)
            X_train_prep = fea.preprocess_imgs(X_train)
            X_test_prep = fea.preprocess_imgs(X_test)
            pbar1.set_postfix({'state': 'Preprocessed'}, refresh=True)

            with tqdm(total=len(classifiers), position=0, leave=False) as pbar2:
                for cls in classifiers:
                    #print(vars(cls))
                    pbar2.set_description(f"Classifying: {cls.__class__.__name__}")
                    y_test, y_pred, report_test, ti = test_single_classifier(cls, fea, X_train_prep, X_test_prep, y_train, y_test)
                    pbar2.set_postfix({'state': 'Save results'}, refresh=True)
                    save_result(cls, fea, X_test_prep, y_test, report_test, ti )
                    pbar2.update()
            pbar1.update()
if __name__ == "__main__":
    test_classifiers(1,1)