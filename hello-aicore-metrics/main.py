import os
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import KFold, train_test_split
from sklearn.inspection import permutation_importance
from datetime import datetime
import pandas as pd
from ai_core_sdk.models import Metric, MetricTag, MetricCustomInfo, MetricLabel
#
from ai_core_sdk.tracking import Tracking
aic_connection = Tracking()
#
# Variables
DATA_PATH = '/app/data/train.csv'
DT_MAX_DEPTH= int(os.getenv('DT_MAX_DEPTH'))
MODEL_PATH = '/app/model/model.pkl'
#
# Load Datasets
df = pd.read_csv(DATA_PATH)
X = df.drop('target', axis=1)
y = df['target']
#
# Metric Logging: Basic
aic_connection.log_metrics(
    metrics = [
        Metric(
            name= "N_observations", value= float(df.shape[0]), timestamp=datetime.utcnow()),
    ]
)
#
# Partition into Train and test dataset
train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.3)
#
# K-fold
kf = KFold(n_splits=5, random_state=31, shuffle=True)
i = 0 # storing step count
for train_index, val_index in kf.split(train_x):
    i += 1
    # Train model on subset
    clf = DecisionTreeRegressor(max_depth=DT_MAX_DEPTH, random_state=31)
    clf.fit(train_x.iloc[train_index], train_y.iloc[train_index])
    # Score on validation data (hold-out dataset)
    val_step_r2 = clf.score(train_x.iloc[val_index], train_y.iloc[val_index])
    # Metric Logging: Step Information
    aic_connection.log_metrics(
        metrics = [
            Metric(name= "(Val) Fold R2", value= float(val_step_r2), timestamp=datetime.utcnow(), step=i),
        ]
    )
    # Delete step model
    del(clf)
#
# Final Model
clf = DecisionTreeRegressor(max_depth=DT_MAX_DEPTH, random_state=31)
clf.fit(train_x, train_y)
# Scoring over test data
test_r2_score = clf.score(test_x, test_y)
# Metric Logging: Attaching to metrics to generated model
aic_connection.log_metrics(
    metrics = [
        Metric(
            name= "Test data R2",
            value= float(test_r2_score),
            timestamp=datetime.utcnow(),
            labels= [
                MetricLabel(name="metrics.ai.sap.com/Artifact.name", value="housepricemodel")
            ]
        )
    ]
)
#
# Model Explaination
r = permutation_importance(
    clf, test_x, test_y,
    n_repeats=30,
    random_state=0
)
# Feature importances
feature_importances = str('')
for i in r.importances_mean.argsort()[::-1]:
    feature_importances += f"{df.columns[i]}: {r.importances_mean[i]:.3f} +/- {r.importances_std[i]:.3f} \n"
# Metric Logging: Custom Structure
aic_connection.set_custom_info(
    custom_info= [
        MetricCustomInfo(name= "Feature Importance (verbose)", value=  str(r)),
        MetricCustomInfo(name= "Feature Importance (brief)", value= feature_importances )
    ]
)
#
# Save model
import pickle
pickle.dump(clf, open(MODEL_PATH, 'wb'))
#
# Metric Logging: Tagging the execution
aic_connection.set_tags(
    tags= [
        MetricTag(name="Validation Method Used", value= "K-Fold"), # your custom name and value
        MetricTag(name="Metrics", value= "R2"),
    ]
)
