import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

df=pd.read_csv('archive (1).zip')

print("shape:", df.shape)
print("\nFrist 5 rows")
df.head()

print("column names:", df.columns.tolist())
print("\nData types:\n", df.dtypes)
print("\nNull vales:\n", df.isnull().sum())

print(df['RiskLevel'].value_counts())
print("\nAs percentage:")
print(df['RiskLevel'].value_counts(normalize=True).round(2)*100)

df.shape

df.describe().round(2)

print("HeartRate below 40:")
print(df[df['HeartRate']<40])
print("\nBodyTemp == 98.0:", (df['BodyTemp'] == 98.0).sum())
print("Total rows:", len(df))
print("Percentage:", round((df['BodyTemp'] == 98.0).sum() /len(df) * 100,1), "%")

df = df.drop_duplicates()
print("after removing duplicates:",df.shape)

df['HeartRate'] = df['HeartRate'].replace(7,np.nan)
df['HeartRate'] = df['HeartRate'].fillna(df['HeartRate'].median())
print("HeartTemp value counts (top 5):")
print(df['BodyTemp'].value_counts().head())

df=pd.read_csv('archive (1).zip')

print("Fresh shape:", df.shape)
print("\nTotal duplicates:", df.duplicated().sum())
print("\nDuplicate rows sample:")
df[df.duplicated(keep=False)].sort_values('Age').head(10)

df=pd.read_csv('archive (1).zip')

df['HeartRate'] = df['HeartRate'].replace(7,np.nan)
df['HeartRate'] = df['HeartRate'].fillna(df['HeartRate'].median())

print("Shape:", df.shape)
print("HeartRate min:", df['HeartRate'].min())
print("HeartRate max:", df['HeartRate'].max())
print("\nAny nulls?", df.isnull().sum().sum())

fig, axes = plt.subplots(2,3, figsize=(10,8))
features_list = ['Age', 'SystolicBP', 'DiastolicBP', 'BS','BodyTemp', 'HeartRate']

for i, feature in enumerate(features_list):
  row, col = i//3,i % 3
  axes[row, col].hist(df[feature], bins=30, color='steelblue', edgecolor='black')
  axes[row, col].set_title(f'Distribution of {feature}')
  axes[row, col].set_xlabel(feature)
  axes[row, col].set_ylabel('Count')
plt.tight_layout()
plt.savefig('feature_distriution.png', dpi=150)
plt.show()
print("cell 6 done")

fig, axes = plt.subplots(2,3, figsize=(10,8))

order = ['low risk', 'mid risk', 'high risk']
color = ['green', 'orange','red']
features_list = ['Age', 'SystolicBP', 'DiastolicBP', 'BS','BodyTemp', 'HeartRate']

for i, feature in enumerate(features_list):
  row,col = i // 3, i % 3
  sns.boxplot(data=df, x='RiskLevel', y=feature, order=order, palette=color, hue='RiskLevel', legend=False, ax=axes[row, col])
  axes[row, col].set_title(f'{feature} vs Risk Level')
  axes[row, col].set_xlabel(' ')

plt.suptitle('How Each Feature Varies Across Risk Levels', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('features_vs_risk.png', dpi=150)
plt.show()
print("cell 7 done")

plt.figure(figsize=(8,6))
sns.heatmap(df.drop('RiskLevel', axis=1).corr(),
            annot=True, fmt='.2f', cmap='coolwarm',
            center=0)
plt.title('Feature COrrelation Matrix')
plt.savefig('correlation_heatmap.png', dpi=150)
plt.show()
print("cell 8 done")

risk_mapping = {'low risk': 0, 'mid risk':1, 'high risk':2}

df['RiskLevel'] = df['RiskLevel'].map(risk_mapping)

print("Encoder Risk Level:")
print(df['RiskLevel'].value_counts())

X = df.drop('RiskLevel',axis=1)
y = df['RiskLevel']
print("Feature Shape:",X.shape)
print("Target Shape",y.shape)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)
print("Traing data shape:",X_train.shape, y_train.shape)
print("Testing data shape:",X_test.shape, y_test.shape)

from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier(random_state=42)

rf_model.fit(X_train, y_train)

y_re = rf_model.predict(X_test)

from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

accuracy = accuracy_score(y_test, y_re)
print(f"Model Accuracy: {accuracy * 100:2f}%")
cm = confusion_matrix(y_test, y_re)
plt.figure(figsize=(8,6))

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Low', 'Mid', 'High'],
            yticklabels=['Low', 'Mid', 'High'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted Risk')
plt.ylabel('actual Risk')
plt.savefig('confusion_matrix.png', dpi=150)
plt.show()

# 1. Extract feature importances
importances = rf_model.feature_importances_

# 2. Create a DataFrame to organize them
feature_importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

# 3. Plot the importances
plt.figure(figsize=(10, 6))
sns.barplot(data=feature_importance_df, x='Importance', y='Feature', palette='viridis')
plt.title('What Drives Maternal Risk? (Feature Importances)')
plt.xlabel('Importance Score (How much influence it has)')
plt.ylabel('Features')
plt.tight_layout()
plt.savefig('feature_importances.png', dpi=150)
plt.show()

!pip install shap

import shap
explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X_test)
plt.figure(figsize=(10,6))
shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
plt.title('SHAP Feature Importance by Risk Level')
plt.savefig('shap_summary_bar.png', dpi=150, bbox_inches='tight')
plt.show()

