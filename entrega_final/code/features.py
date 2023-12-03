import abc
import numpy as np
from skimage import feature
from skimage.transform import integral_image

from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

class Features(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def preprocess_img(self, img : np.ndarray):
        pass

    @abc.abstractmethod
    def preprocess_imgs(self, images: np.ndarray):
        res = []
        with ProcessPoolExecutor() as executor:
            res = list(tqdm(executor.map(self.preprocess_img, images), total=len(images), desc='Preprocessing Images ' + self.__class__.__name__, unit="image"))
        return res

class Reshape(Features):  
    def __init__(self) -> None:
        super().__init__()
    
    def preprocess_img(self, img : np.ndarray):
        x, y = np.shape(img)
        return np.reshape(img, (1, x * y))

    def preprocess_imgs(self, images: np.ndarray):
        x, y = np.shape(images[0])
        return np.reshape(images , (len(images), x * y))

class HOGPrepocess(Features):
    def __init__(self) -> None:
        super().__init__()
    
    def preprocess_img(self, img : np.ndarray):
        return feature.hog(img)
        
    def preprocess_imgs(self, images: np.ndarray):
        return super().preprocess_imgs(images)
    
class HAARPreprocess(Features):
    def __init__(self) -> None:
        super().__init__()

    def extract_feature_image(self, img):
        feature_coord, feature_type = self.roi_haar()

        feature_coord = np.concatenate([x[::2] for x in feature_coord])
        feature_type = np.concatenate([x[::2] for x in feature_type])
        """Extract the haar feature for the current image"""
        ii = integral_image(img)
        return feature.haar_like_feature(ii, 0, 0, img.shape[0], img.shape[1],
                             feature_type=feature_type,
                             feature_coord=feature_coord)
    
    def preprocess_img(self, img: np.ndarray):
        feature_types = ['type-2-x']#, 'type-2-y']# ,'type-3-x', 'type-3-y']#, 'type-4']
        return self.extract_feature_image(img)

    def roi_haar(self):
        return feature.haar_like_feature_coord(62, 47, ['type-2-x'])

    def preprocess_imgs(self, images: np.ndarray):
        return super().preprocess_imgs(images)
