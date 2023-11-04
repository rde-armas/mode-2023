from sklearn.datasets import fetch_lfw_people
from sklearn.feature_extraction.image import PatchExtractor
from skimage.transform import resize
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray

IMG_PATH = './imagenes/'

def load_img(path):
    img_background = []
    for i in range(16):
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

def negative_patches(path = IMG_PATH):
    def to_grayscale(img):
        return rgb2gray(img)
    shape = positive_patches()[1].shape
    img_backgroud = load_img(path)
    background_grayscale = list(map(to_grayscale, img_backgroud))
    patches = [extract_patches(back, shape) for back in background_grayscale]
    background_shapes = np.reshape(patches, (16 * 820, 62, 47))
    return background_shapes


