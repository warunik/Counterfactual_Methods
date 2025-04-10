import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn import model_selection
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from Foil_Trees import domain_mappers, contrastive_explanation, explanators

SEED = np.random.RandomState(1994)

diabetes = load_diabetes()

#dataset list
# df = pd.DataFrame(data=diabetes.data, columns=diabetes.feature_names)
# df['target'] = diabetes.target
# print("Diabetes dataset:")
# print(df.head())

rx_train, rx_test, ry_train, ry_test = model_selection.train_test_split(diabetes.data, 
                                                                        diabetes.target, 
                                                                        train_size=0.80, 
                                                                        random_state=SEED)

m_cv = RandomForestRegressor(random_state=SEED)
r_model = model_selection.GridSearchCV(m_cv, cv=5, param_grid={'n_estimators': [50, 100, 500]})
r_model.fit(rx_train, ry_train)

print('Regressor performance (R-squared):', metrics.r2_score(ry_test, r_model.predict(rx_test)))

r_sample = rx_test[1]
print('\nFeature names:', diabetes.feature_names)
print(r_sample)

r_dm = contrastive_explanation.DomainMapperTabular(rx_train, 
                                             feature_names=diabetes.feature_names)

r_exp = contrastive_explanation.ContrastiveExplanation(r_dm,
                                  regression=True,
                                  explanator=contrastive_explanation.TreeExplanator(verbose=True),
                                  verbose=False)

print("\nExplanation:", r_exp.explain_instance_domain(r_model.predict, r_sample, include_factual=True))