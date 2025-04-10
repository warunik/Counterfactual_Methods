
import numpy as np
from sklearn import metrics
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

import Foil_Trees.domain_mappers as map

SEED = np.random.RandomState(1994)

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3, random_state=SEED)

dm = map.DomainMapperTabular(
    train_data=X_train,
    feature_names=iris.feature_names,
    contrast_names=iris.target_names.tolist()
)

model = RandomForestClassifier(random_state=SEED).fit(X_train, y_train)

print('Classifier performance (F1):', metrics.f1_score(y_test, model.predict(X_test), average='weighted'))