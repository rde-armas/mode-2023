import clasifiers as clf
import features as feat
import carga as carga
import numpy as np
from sklearn.model_selection import train_test_split
import random
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score, roc_curve, auc

import pandas as pd

import time
from tqdm import tqdm
import joblib
from tabulate import tabulate

def test_classifier(classifier: clf.Classifier, fea: feat.Features, X_train, X_test, y_train, y_test):
     # va a faltar agregar el nombre de los parametros de los modelos

    train_st = time.time()
    classifier.train(X_train, y_train)
    joblib.dump(classifier, f'./model/{classifier.__class__.__name__}{fea.__class__.__name__}.pkl')
    print('Tiempo entrenamiento', (time.time() - train_st))
    train_et = time.time()
    y_pred = classifier.classify(X_test)
    report_test = classification_report(y_test, y_pred, output_dict=True)

    return y_test, y_pred, report_test, train_et - train_st


# Guarda las imagenes despues de sacar las features 
def save_imgs_matrix(X: np.array, feature, des: str, batch_size: int = 100):
    filename = f'./data/{feature.__class__.__name__}_{des}.npz'
    matrices = {}

    for i in tqdm(range(0, X.shape[0], batch_size), desc='Guardando matrices en lotes'):
        batch_indices = range(i, min(i + batch_size, X.shape[0]))
        batch_matrices = {str(idx): feature.preprocess_img(X[idx]) for idx in batch_indices}

        try:
            with np.load(filename) as data:
                # Cargar datos existentes
                matrices = {f'{key}': value for key, value in data.items()}
        except FileNotFoundError:
            matrices = {}  # Si el archivo no existe, iniciar con un diccionario vacío

        # Agregar las nuevas matrices con índices incrementales
        matrices.update(batch_matrices)

        # Guardar todas las matrices en el archivo
        np.savez(filename, **matrices)

    return

# Guarda los resultados de los experimentos
def save_result(cls: clf.Classifier , fea: feat.Features, model, X_test, y_test, report_test, ti):
    headers = ["Classifier", "Preprocessing", "Accuracy", "Precision", "Recall/TPR", "FPR", "F1-Score", "ROC curve (area)", "Balanced Accuracy", "Time Train"]
    
    conf_matrix = confusion_matrix(y_test, y_pred)
    tp, fp, fn, tn = conf_matrix.ravel()
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)

    roc_auc = auc(fpr, tpr)
    y_pred = model.classify(X_test)
    tp / (tp + fn),
    fp / (tn + fp),

    # Crear un diccionario con los datos
    data = {
        "Classifier": [cls.__class__.__name__],
        "Preprocessing": [fea.__class__.__name__],
        "Accuracy": [report_test['accuracy']],
        "Precision": [report_test['1']['precision']],
        "Recall/TPR": [tp / (tp + fn)],
        "FPR": [fp / (tn + fp)],
        "F1-Score": [report_test['1']['f1-score']],
        "ROC curve (area)": [roc_auc],
        "Balanced Accuracy": [balanced_accuracy_score(y_test,y_pred)],
        "Time Train": [ti]
    }

    # Crear el DataFrame
    df = pd.DataFrame(data, columns=headers)

    return 


def testing(pro_train: int, prop_test: int, test_size_positive = 0.1):

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
    print(len(samples_test))
    print(len(samples_train))

    
    X_train, _ , y_train, _ = train_test_split(samples_train, labels_train, test_size = 0.00001)
    _, X_test, _, y_test = train_test_split(samples_test, labels_test, test_size = 0.99999)

    reshape = feat.Reshape() 
    hog = feat.HOGPrepocess()
    haar = feat.HAARPreprocess()

    save_imgs_matrix(X_train, reshape, 'train')
    save_imgs_matrix(X_train, hog, 'train')
    save_imgs_matrix(X_train, haar, 'train')
    save_imgs_matrix(X_train, reshape, 'test')
    save_imgs_matrix(X_train, hog, 'test')
    save_imgs_matrix(X_train, haar, 'test')

    # knn_classifier = clf.KNNClassifier()
    # dtree_classifier = clf.DTreeClassifier()
    # logistic_regression_classifier = clf.LogisticRegressionClassifier()
    # rf_classifier = clf.RFClassifier()

    # classifiers = [knn_classifier, logistic_regression_classifier,dtree_classifier, rf_classifier]

    # results = []
    


if __name__ == "__main__":
    print(testing(5,100))
    # positive_faces = carga.positive_patches()
    # negative_faces = carga.negative_patches('./imagenes/')

    # # Etiquetas para caras (clase positiva)
    # positive_labels = np.tile(1, 13233)

    # # Etiquetas para fondos (clase negativa)
    # negative_labels = np.tile(0, 13120)

    # samples = np.concatenate((positive_faces, negative_faces))
    # labels = np.concatenate((positive_labels, negative_labels))
    # X_train, X_test, y_train, y_test = train_test_split(samples, labels, test_size=0.20, random_state=42)

    # # X_train = X_train[10:100]
    # # y_train = y_train[10:100]

    # # X_test = X_test[10:30]
    # # y_test = y_test[10:30]

    # features_methods = [ feat.HOGPrepocess(), feat.HAARPreprocess()]
    # #podemos pasar parametros si es necesario

    # knn_classifier = clf.KNNClassifier()
    # dtree_classifier = clf.DTreeClassifier()
    # logistic_regression_classifier = clf.LogisticRegressionClassifier()
    # rf_classifier = clf.RFClassifier()

    # classifiers = [knn_classifier, logistic_regression_classifier,dtree_classifier, rf_classifier]

    # results = []
    # for fea in features_methods:
    #     X_train_prep = fea.preprocess_imgs(X_train)
    #     X_test_prep = fea.preprocess_imgs(X_test)
    #     for cls in classifiers:
    #         y_test, y_pred, report_test, ti = test_classifier(cls, fea, X_train_prep, X_test_prep, y_train, y_test)
    #         conf_matrix = confusion_matrix(y_test, y_pred)
    #         fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    #         roc_auc = auc(fpr, tpr)
    #         tp, fp, fn, tn = conf_matrix.ravel()
    #         #print(f"Testing {cls.__class__.__name__} con {prep.__class__.__name__} {report_test['accuracy']} {report_test['macro avg']['precision']} {report_test['macro avg']['recall']} {report_test['macro avg']['f1-score']} {ti}")
    #         results.append([
    #             cls.__class__.__name__,
    #             fea.__class__.__name__,
    #             report_test['accuracy'],
    #             report_test['1']['precision'],
    #             tp / (tp + fn),
    #             fp / (tn + fp),
    #             roc_auc,
    #             balanced_accuracy_score(y_test,y_pred),
    #             report_test['1']['f1-score'],
    #             ti
    #         ])  
    #         #print(f"Precision: {report_test}")
    # headers = ["Classifier", "Preprocessing", "Accuracy", "Precision", "Recall/TPR", "FPR", "F1-Score", "ROC curve (area)", "Balanced Accuracy", "Time Train"]
    # print(tabulate(results, headers, tablefmt="grid"))

  