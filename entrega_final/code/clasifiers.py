import abc
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

Metaparameters = list[(str, object)]

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

class LogisticRegressionClassifier(Classifier):
    def __init__(self, parameters="") -> None:
        super().__init__(parameters)
        self.classifier = LogisticRegression(solver='liblinear', random_state=42)
    
    def classify(self, images: list[np.ndarray]) -> list[int]:
        return self.classifier.predict(images)

    def train(self, X_train, Y_train):
        self.classifier.fit(X_train, Y_train)
    
class KNNClassifier(Classifier):
    def __init__(self, metaparams: Metaparameters) -> None:
        super().__init__(metaparams)
        nn = self.mp[0][1]
        weight = self.mp[1][1]
        metric_exponent = self.mp[2][1]
        self.classifier = KNeighborsClassifier(n_neighbors=nn, weights=weight, p=metric_exponent,n_jobs=-1)
    
    def classify(self, images: list[np.ndarray])->list[int]:
        return self.classifier.predict(images)
    
    def train(self, X_train, Y_train):
        self.classifier.fit(X_train, Y_train)

class RFClassifier(Classifier):
    def __init__(self, parameters="") -> None:
        super().__init__(parameters)
        self.classifier = RandomForestClassifier()
    
    def classify(self, images: list[np.ndarray]) -> list[int]:
        return self.classifier.predict(images)
    
    def train(self, X_train, Y_train):
        self.classifier.fit(X_train, Y_train)

class DTreeClassifier(Classifier):
    def __init__(self, parameters="") -> None:
        super().__init__(parameters)
        self.classifier = DecisionTreeClassifier()
    
    def classify(self, images: list[np.ndarray]) -> list[int]:
        return self.classifier.predict(images)
    
    def train(self, X_train, Y_train):
        self.classifier.fit(X_train, Y_train)
    