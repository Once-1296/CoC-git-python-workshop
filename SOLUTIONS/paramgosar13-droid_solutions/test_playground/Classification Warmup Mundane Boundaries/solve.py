import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import log_loss
from pathlib import Path

script_dir = Path(__file__).resolve().parent
data_dir = script_dir / "data"
train_file = data_dir / "train_1.csv"
test_file = data_dir / "test_1.csv"
submission_file = data_dir / "submission.csv"

train_df = pd.read_csv(train_file)
X = train_df.drop(columns=['ID', 'Class'])
y = train_df['Class']

print(X.head())
X.info()
print(X.isnull().sum())
print(X.describe())

plt.figure(figsize=(10,6))
sns.scatterplot(data=train_df, x="X", y="Y", hue="Class", palette="Set2")
plt.title("Box Plot of Y vs X (Cloured by Class)")
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", RandomForestClassifier(random_state=0))
]) 

neg_cv_log_loss_score = cross_val_score(pipeline, X, y, cv=5, scoring='neg_log_loss')
cv_log_loss_score = -(neg_cv_log_loss_score)
print(f"\nLog Loss scores: {cv_log_loss_score}")
print(f"Mean Log Loss scores: {cv_log_loss_score.mean():.4f}")
print(f"SD of Log Loss scores: {cv_log_loss_score.std():.4f}")

pipeline.fit(X_train,y_train)

y_pred_ll = pipeline.predict_proba(X_test)

loss = log_loss(y_test, y_pred_ll)

print(f"\nLog Loss: {loss:.4f}")

test_df = pd.read_csv(test_file)
test = test_df.drop(columns=['ID'])
predict = pipeline.predict_proba(test)
submission = pd.DataFrame({
    'ID': test_df['ID'],
    'Class': predict[:,-1]
})

submission.to_csv(submission_file, index=False)
