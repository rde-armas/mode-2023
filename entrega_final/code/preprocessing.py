import abc
import numpy as np
from skimage import feature

class Preprocess(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def preprocess_img(self, img : np.ndarray):
        pass

    
    @abc.abstractmethod
    def preprocess_imgs(self, images: np.ndarray):
        res = []
        for x in images:
            res.append(self.preprocess_img(x))
        return res

class reshape(Preprocess):  
    def __init__(self) -> None:
        super().__init__()
    
    def preprocess_img(self, img : np.ndarray):
        x, y = np.shape(img)
        return np.reshape(img, (1, x * y))

    def preprocess_imgs(self, images: np.ndarray):
        x, y = np.shape(images[0])
        return np.reshape(images , (len(images), x * y))

class HOGPrepocess(Preprocess):
    def __init__(self) -> None:
        super().__init__()
    
    def preprocess_img(self, img : np.ndarray):
        return feature.hog(img)
        
    def preprocess_imgs(self, images: np.ndarray):
        return super().preprocess_imgs(images)
    
class HAARPreprocess(Preprocess):
    def __init__(self) -> None:
        super().__init__()
    
    def preprocess_img(self, img: np.ndarray):
        return img

    def preprocess_imgs(self, images: np.ndarray):
        return super().preprocess_imgs(images)
    
class AllPreprocess(Preprocess):
        def __init__(self) -> None:
            super().__init__()
        
        def preprocess_img(self, img: np.ndarray):
            return img
    
        def preprocess_imgs(self, images: np.ndarray):
            return super().preprocess_imgs(images)