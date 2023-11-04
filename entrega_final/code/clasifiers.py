import abc
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


class Classifier(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def __init__(self, parameters):
        self.mp = parameters
        pass
    @abc.abstractmethod
    def classify(sefl, images : np.ndarray):
        pass
    
    @abc.abstractmethod
    def train(self, X_train, Y_train):
        pass

class LogisticRegression(Classifier):
    def __init__(self, parameters) -> None:
        super().__init__(parameters)
        self.classifier = LogisticRegression(solver='liblinear', random_state=42)
        pass
    
    def classify(self, images: list[np.ndarray]) -> list[int]:
        return self.classifier.predict(images)

    def train(self, X_train, Y_train) -> np.ndarray:
        return self.classifier.fit(X_train, Y_train)
    
class KNeighborsClassifier(Classifier):
    def __init__(self, parameters) -> None:
        super().__init__(parameters)
        self.classifier = KNeighborsClassifier()
    
    def classify(self, images: list[np.ndarray])->list[int]:
        return self.classifier.predict(images)
    
    def train(self, X_train, Y_train):
        return self.classifier.fit(X_train, Y_train)

class RandomForestClassifier(Classifier):
    def __init__(self, parameters) -> None:
        super().__init__(parameters)
        self.classifier = RandomForestClassifier()
    
    def classify(self, images: list[np.ndarray]) -> list[int]:
        return self.classifier.predict(images)
    
    def train(self, X_train, Y_train):
        return self.classifier.fit(X_train, Y_train)

class DecisionTreeClassifier(Classifier):
    def __init__(self, parameters) -> None:
        super().__init__(parameters)
        self.classifier = DecisionTreeClassifier()
    
    def classify(self, images: list[np.ndarray]) -> list[int]:
        return self.classifier.predict(images)
    
    def train(self, X_train, Y_train):
        return super().fit(X_train, Y_train)