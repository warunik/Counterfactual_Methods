import numpy as np
from sklearn import metrics, model_selection
from sklearn.datasets import load_diabetes
from sklearn.neighbors import KNeighborsRegressor
from Foil_Trees import domain_mappers, contrastive_explanation

SEED = np.random.RandomState(1994)

diabetes = load_diabetes()
X_train, X_test, y_train, y_test = model_selection.train_test_split(
    diabetes.data, diabetes.target, train_size=0.80, random_state=SEED)

model = model_selection.GridSearchCV(
    KNeighborsRegressor(),
    param_grid={'n_neighbors': [3, 5, 7], 'weights': ['uniform', 'distance']},
    cv=5
)
model.fit(X_train, y_train)

print('Decision Tree R²:', metrics.r2_score(y_test, model.predict(X_test)))

dm = domain_mappers.DomainMapperTabular(X_train, feature_names=diabetes.feature_names)
exp = contrastive_explanation.ContrastiveExplanation(dm, regression=True,
                                  explanator=contrastive_explanation.TreeExplanator(),
                                  verbose=False)

print("\nExplanation:", exp.explain_instance_domain(model.predict, X_test[1]))