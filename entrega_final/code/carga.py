from sklearn.datasets import fetch_lfw_people
from sklearn.feature_extraction.image import PatchExtractor
from skimage.transform import resize
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
import random
from sklearn.model_selection import train_test_split

IMG_PATH = './imagenes/'

def load_img(path: str):
    img_background = []
    for i in range(30):
        with open(f'{path}{i}.jpg', 'rb') as background:
            img_background.append(plt.imread(background))
    return img_background

def extract_patches(img, patch_shape, N=820, scale=1.0):
    # Calcula el tamaño del parche extraído basado en el factor de escala dado
    extracted_patch_size = tuple((scale * np.array(patch_shape)).astype(int))

    # Inicializa un objeto PatchExtractor con el tamaño de parche calculado,
    # el número máximo de parches, y una semilla de estado aleatorio
    extractor = PatchExtractor(patch_size=extracted_patch_size, max_patches=N, random_state=0)

    # Extrae parches de la imagen dada
    # img[np.newaxis] se utiliza la entrada de PatchExtractor es un conjunto de imágenes
    patches = extractor.transform(img[np.newaxis])

    # Si el factor de escala no es 1, redimensiona cada parche extraído
    # al tamaño del parche original
    if scale != 1:
        patches = np.array([resize(patch, patch_shape) for patch in patches])

    # Devuelve la lista de parches extraídos (y posiblemente redimensionados)
    return patches

def positive_patches():
    faces = fetch_lfw_people()
    faces_img = faces.images
    return faces_img

def negative_patches(N: int, path = IMG_PATH):
    N = int(N / 30)
    def to_grayscale(img):
        return rgb2gray(img)
    shape = positive_patches()[1].shape
    img_backgroud = load_img(path)
    background_grayscale = list(map(to_grayscale, img_backgroud))
    patches = [extract_patches(back, shape, N) for back in background_grayscale]

    background_shapes = np.reshape(patches, (30 * N, 62, 47))
    return background_shapes

def get_train_test(pro_train: int, prop_test: int, test_size_positive = 0.1):
        # Cargar rostros
    positive_faces = positive_patches()
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
    negative_faces_train = negative_patches(train_amount)
    negative_faces_test = negative_patches(test_amount)

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

    return X_train, X_test, y_train, y_test