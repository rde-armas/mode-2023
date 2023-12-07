import abc
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
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

    def get_metaparameters(self):
        if self.mp is None or len(self.mp) == 0:
            return ''
        return ", ".join(map(lambda x: ": ".join(map(str, x)), self.mp))

class LogisticRegressionClassifier(Classifier):
    def __init__(self, metaparams: Metaparameters) -> None:
        super().__init__(metaparams)
        penalty = self.mp[0][1]
        solver = self.mp[1][1]
        max_iter = self.mp[2][1]
        multi_class = self.mp[3][1]
        l1_ratio = self.mp[4][1]
        self.classifier = LogisticRegression(
            penalty=penalty, solver=solver, max_iter=max_iter, 
            multi_class=multi_class,l1_ratio=l1_ratio, random_state=42
            )
    
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
    def __init__(self, metaparams: Metaparameters) -> None:
        super().__init__(metaparams)
        n_estimators = self.mp[0][1]
        criterion = self.mp[1][1]
        max_depth = self.mp[2][1]
        min_samples_split = self.mp[3][1]
        max_features = self.mp[4][1]
        self.classifier = RandomForestClassifier(
            n_estimators=n_estimators, criterion=criterion, max_depth=max_depth, 
            min_samples_split=min_samples_split, max_features=max_features
            )
    
    def classify(self, images: list[np.ndarray]) -> list[int]:
        return self.classifier.predict(images)
    
    def train(self, X_train, Y_train):
        self.classifier.fit(X_train, Y_train)

class DTreeClassifier(Classifier):
    def __init__(self, metaparams: Metaparameters) -> None:
        super().__init__(metaparams)
        criteion = self.mp[0][1]
        max_depth = self.mp[1][1]
        min_samples_split = self.mp[2][1]
        min_samples_leaf = self.mp[3][1]
        max_features = self.mp[4][1]
        self.classifier = DecisionTreeClassifier(
            criterion=criteion, max_depth=max_depth, min_samples_leaf=min_samples_leaf,
            min_samples_split=min_samples_split, max_features=max_features
            )
    
    def classify(self, images: list[np.ndarray]) -> list[int]:
        return self.classifier.predict(images)
    
    def train(self, X_train, Y_train):
        self.classifier.fit(X_train, Y_train)
    
class GBoostingClassifier(Classifier):
    def __init__(self, metaparams: Metaparameters) -> None:
        super().__init__(metaparams)
        n_estimators = self.mp[0][1]
        loss = self.mp[1][1]
        max_depth = self.mp[2][1]
        min_samples_split = self.mp[3][1]
        max_features = self.mp[4][1]
        self.classifier = GradientBoostingClassifier(
            n_estimators=n_estimators, loss=loss,  max_depth=max_depth, 
            min_samples_split=min_samples_split, max_features=max_features
            )
    
    def classify(self, images: list[np.ndarray]) -> list[int]:
        return self.classifier.predict(images)
    
    def train(self, X_train, Y_train):
        self.classifier.fit(X_train, Y_train)
