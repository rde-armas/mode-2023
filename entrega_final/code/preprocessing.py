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
            res.append(self.preprocess_tweet(x))
        return res
    
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

        return None

    def preprocess_imgs(self, images: np.ndarray):
        return super().preprocess_imgs(images)