import dice_ml
from dice_ml.utils import helpers
import joblib  # Needed to load the pre-trained model

from sklearn.ensemble import RandomForestClassifier
from dice_ml.utils import helpers

data = helpers.load_adult_income_dataset()
X = data.drop('income', axis=1)
y = data['income']

# Train fresh model
model = RandomForestClassifier().fit(X, y)

# Use with DiCE
d = dice_ml.Data(dataframe=data,
                continuous_features=['age', 'hours_per_week'],
                outcome_name='income')
m = dice_ml.Model(model=model, backend='sklearn')

# Generate counterfactuals
exp = dice_ml.Dice(d, m)
query_instance = data.drop('income', axis=1).iloc[0:1]
cf = exp.generate_counterfactuals(query_instance, total_CFs=3, desired_class="opposite")
cf.visualize_as_dataframe()