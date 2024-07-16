
# import important libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, confusion_matrix, roc_auc_score, roc_curve
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler


# Loading dataset
data = pd.read_csv("copd.csv")

# Initial data checks
print(data.head())
print(data.shape)
print(data.info())
print(data.isnull().sum())
print(data.duplicated().sum())
print(data.describe(include=np.number).T)

data_object = data.select_dtypes(include=['object']).columns
for col in data_object:
    value_counts = data[col].value_counts()
    print(f"Value counts for '{col}':")
    print(value_counts)
    print()

df = data.copy()
df = df.drop('COPD_control', axis=1)

# Plot histograms
sns.histplot(df, x='packyrs_10', kde=True)
plt.title('Histogram for pack years of smoking')
plt.xlabel('Pack years of smoking')
plt.savefig('hist_packyrs_10.png')
plt.clf()

sns.histplot(df, x='wholelung950', kde=True)
plt.title('Histogram for Lung area')
plt.xlabel('Lung area')
plt.savefig('hist_wholelung950.png')
plt.clf()

# Plot bar graphs
data_object_copd = df.select_dtypes(include=['object']).columns
for col in data_object_copd:
    sns.countplot(data=df, x=col, palette='bright')
    plt.xlabel(col)
    plt.ylabel('frequency')
    plt.title(f'bar graph for {col}')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'bar_{col}.png')
    plt.clf()

# plot pair plot
sns.pairplot(df, hue='copd_exacerb_cat', diag_kind='kde')
plt.savefig('pairplot.png')
plt.clf()

# plot grouped bar graphs
for col in data_object_copd:
    sns.countplot(x=col, data=df, hue='copd_exacerb_cat', palette='bright')
    plt.xlabel(col)
    plt.ylabel('frequency')
    plt.title(f'bar graph for {col}')
    plt.xticks(rotation=45)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.savefig(f'bar_{col}_hue.png')
    plt.clf()

# Handle missing values
for column in df.columns:
    if df[column].dtype == 'object':
        df[column].fillna(df[column].mode()[0], inplace=True)
    else:
        df[column].fillna(df[column].median(), inplace=True)
# recheck missingness
df.isnull().sum()

# Convert categorical variables
categorical_columns = [
    "sex", "age_cat2", "bodycomp", "smok_habits", "diabetes", "statin",
    "ARB_ACE_all", "sign_CACS", "cor_stenosis", "gold", "copd_exacerb_cat",
    "resp_failure", "eosinophilic_COPD", "crp_cat"
]
for col in categorical_columns:
    df[col] = df[col].astype("category")

# one hot encoding
df = pd.get_dummies(df, drop_first=True)
bool_columns = df.select_dtypes(include=['bool']).columns
df[bool_columns] = df[bool_columns].astype(int)

# Define features and target
X = df.drop('copd_exacerb_cat_yes', axis=1)
y = df['copd_exacerb_cat_yes']

# split data to training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# scale the numeric columns
scaler = StandardScaler()
X_train[['packyrs_10', 'wholelung950']] = scaler.fit_transform(X_train[['packyrs_10', 'wholelung950']])
X_test[['packyrs_10', 'wholelung950']] = scaler.fit_transform(X_test[['packyrs_10', 'wholelung950']])

# Logistic Regression
logreg = LogisticRegression(random_state=42)
logreg.fit(X_train, y_train)
y_pred_logreg = logreg.predict(X_test)
print("Logistic Regression on Original Data")
print(classification_report(y_test, y_pred_logreg))
cm = confusion_matrix(y_test, y_pred_logreg)
ConfusionMatrixDisplay(cm).plot()
plt.title("Logistic Regression - Original Data")
plt.savefig('logreg_cm_original.png')
plt.clf()

# Decision Tree
tree = DecisionTreeClassifier(random_state=42)
tree.fit(X_train, y_train)
y_pred_tree = tree.predict(X_test)
print("Decision Tree on Original Data")
print(classification_report(y_test, y_pred_tree))
cm = confusion_matrix(y_test, y_pred_tree)
ConfusionMatrixDisplay(cm).plot()
plt.title("Decision Tree - Original Data")
plt.savefig('tree_cm_original.png')
plt.clf()

# Random Forest
forest = RandomForestClassifier(random_state=42)
forest.fit(X_train, y_train)
y_pred_forest = forest.predict(X_test)
print("Random Forest on Original Data")
print(classification_report(y_test, y_pred_forest))
cm = confusion_matrix(y_test, y_pred_forest)
ConfusionMatrixDisplay(cm).plot()
plt.title("Random Forest - Original Data")
plt.savefig('forest_cm_original.png')
plt.clf()

# Apply SMOTE to oversample the minority class in the training set
sm = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = sm.fit_resample(X_train, y_train)

# logistic regression for oversampled data
logreg.fit(X_train_resampled, y_train_resampled)
y_pred_logreg = logreg.predict(X_test)
print("Logistic Regression on Oversampled Data")
print(classification_report(y_test, y_pred_logreg))
cm = confusion_matrix(y_test, y_pred_logreg)
ConfusionMatrixDisplay(cm).plot()
plt.title("Logistic Regression - Oversampled Data")
plt.savefig('logreg_cm_oversampled.png')
plt.clf()

# Decision Tree for oversampled training data
tree.fit(X_train_resampled, y_train_resampled)
y_pred_tree = tree.predict(X_test)
print("Decision Tree on Oversampled Data")
print(classification_report(y_test, y_pred_tree))
cm = confusion_matrix(y_test, y_pred_tree)
ConfusionMatrixDisplay(cm).plot()
plt.title("Decision Tree - Oversampled Data")
plt.savefig('tree_cm_oversampled.png')
plt.clf()

# Random Forest for the oversampled training data
forest.fit(X_train_resampled, y_train_resampled)
y_pred_forest = forest.predict(X_test)
print("Random Forest on Oversampled Data")
print(classification_report(y_test, y_pred_forest))
cm = confusion_matrix(y_test, y_pred_forest)
ConfusionMatrixDisplay(cm).plot()
plt.title("Random Forest - Oversampled Data")
plt.savefig('forest_cm_oversampled.png')
plt.clf()

# Apply RandomUnderSampler
us = RandomUnderSampler(random_state=42)
X_train_under, y_train_under = us.fit_resample(X_train, y_train)

# logistic regression for undersampled training data
logreg.fit(X_train_under, y_train_under)
y_pred_logreg = logreg.predict(X_test)
print("Logistic Regression on Undersampled Data")
print(classification_report(y_test, y_pred_logreg))
cm = confusion_matrix(y_test, y_pred_logreg)
ConfusionMatrixDisplay(cm).plot()
plt.title("Logistic Regression - Undersampled Data")
plt.savefig('logreg_cm_undersampled.png')
plt.clf()

# decision tree for undersampled training data
tree.fit(X_train_under, y_train_under)
y_pred_tree = tree.predict(X_test)
print("Decision Tree on Undersampled Data")
print(classification_report(y_test, y_pred_tree))
cm = confusion_matrix(y_test, y_pred_tree)
ConfusionMatrixDisplay(cm).plot()
plt.title("Decision Tree - Undersampled Data")
plt.savefig('tree_cm_undersampled.png')
plt.clf()

# Random Forest for the undersampled training data
forest.fit(X_train_under, y_train_under)
y_pred_forest = forest.predict(X_test)
print("Random Forest on Undersampled Data")
print(classification_report(y_test, y_pred_forest))
cm = confusion_matrix(y_test, y_pred_forest)
ConfusionMatrixDisplay(cm).plot()
plt.title("Random Forest - Undersampled Data")
plt.savefig('forest_cm_undersampled.png')
plt.clf()

# ROC-AUC for Logistic Regression for undersampled data
logreg.fit(X_train_under, y_train_under)
y_proba_logreg = logreg.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_proba_logreg)
roc_auc = roc_auc_score(y_test, y_proba_logreg)
print(f"ROC-AUC Score for Logistic Regression: {roc_auc:.4f}")
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC Curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Logistic Regression')
plt.legend(loc="lower right")
plt.savefig('roc_logreg.png')
plt.clf()


# ROC-AUC for Decision Tree for undersampled data
tree.fit(X_train_under, y_train_under)
y_proba_tree = tree.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_proba_tree)
roc_auc = roc_auc_score(y_test, y_proba_tree)
print(f"ROC-AUC Score for Decision Tree: {roc_auc:.4f}")
plt.plot(fpr, tpr, color='green', label=f'Decision Tree (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Decision Tree')
plt.legend(loc="lower right")
plt.savefig('roc_tree.png')
plt.clf()


# ROC-AUC for Random Forest for undersampled data
forest.fit(X_train_under, y_train_under)
y_proba_forest = forest.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_proba_forest)
roc_auc = roc_auc_score(y_test, y_proba_forest)
print(f"ROC-AUC Score for Random Forest: {roc_auc:.4f}")
plt.plot(fpr, tpr, color='red', label=f'Random Forest (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Random Forest')
plt.legend(loc="lower right")
plt.savefig('roc_forest.png')
plt.clf()


# Get feature importances
importances = forest.feature_importances_
feature_names = X_train_under.columns 

# Sort features by importance
indices = np.argsort(importances)

# Create a DataFrame for easier handling
feature_importance_df = pd.DataFrame({
    'feature': feature_names[indices],
    'importance': importances[indices]
})

# Plot
plt.figure(figsize=(10, 12))  # Adjusted figure size for better visibility
plt.title("Feature Importances in Random Forest Model")
plt.barh(range(len(importances)), feature_importance_df['importance'])
plt.yticks(range(len(importances)), feature_importance_df['feature'])
plt.xlabel('Importance')
plt.ylabel('Features')
plt.tight_layout()
plt.savefig('feature_importances_forest.png')
plt.clf()

# Print top 10 features
print("Top 5 most important features:")
print(feature_importance_df.iloc[::-1].head(5))  # Reverse order to show most important first
