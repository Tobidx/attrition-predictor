#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[27]:


import pandas as pd
data=pd.read_csv('HR_Analytics.csv')
pd.options.display.max_columns=None
data=data.dropna()

import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from scipy.stats import randint, uniform
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV

# EDA

data.dtypes
df=pd.DataFrame(data)


target='Attrition'
X= df.drop(columns=[target, 'EmpID'], axis=1)
y= df[target]

from sklearn.preprocessing import OneHotEncoder


# one-hot encoding
encoder=OneHotEncoder(sparse_output=False)
categorical_columns=X.select_dtypes(include=['object']).columns.tolist()
one_hot_encoded=encoder.fit_transform(X[categorical_columns])
one_hot_df=pd.DataFrame(one_hot_encoded, columns=encoder.get_feature_names_out(categorical_columns))

# Concatenate the one-hot encoded dataframe with the original dataframe
X_encoded = pd.concat([X, one_hot_df], axis=1)

# Drop the original categorical columns
# After one-hot encoding and before scaling
X_encoded = pd.concat([X, one_hot_df], axis=1)
X_encoded = X_encoded.drop(categorical_columns, axis=1)

# Combine X and y into a single dataframe
combined_df = pd.concat([X_encoded, y], axis=1)

# Drop NaN values from the combined dataframe
combined_df = combined_df.dropna()

# Split back into X and y
X = combined_df.drop(columns=[target])
y = combined_df[target]

# Continue with your mapping of y
y = y.map({'Yes': 1, 'No': 0})


from sklearn.preprocessing import StandardScaler
# Now proceed with scaling
scaler = StandardScaler()
scaled_X = scaler.fit_transform(X)



from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2, random_state=42)


# Compute class weights
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = dict(zip(np.unique(y_train), class_weights))

# Convert to XGBoost's scale_pos_weight format
scale_pos_weight = 2.8
# Define the parameter distribution for random search
param_dist = {
    'max_depth': randint(3, 10),
    'learning_rate': uniform(0.01, 0.3),
    'subsample': uniform(0.5, 0.5),
    'colsample_bytree': uniform(0.5, 0.5),
    'min_child_weight': randint(1, 10),
    'n_estimators': randint(100, 1000),
    'scale_pos_weight': [scale_pos_weight]  # Add this parameter
}

# Create the base model
xgb_model = xgb.XGBClassifier(random_state=42)

# Create the RandomizedSearchCV object
random_search = RandomizedSearchCV(
    estimator=xgb_model,
    param_distributions=param_dist,
    n_iter=100,
    cv=5,
    scoring='f1',
    random_state=42,
    n_jobs=-1
)

# Fit the RandomizedSearchCV object to the training data
random_search.fit(X_train, y_train)

# Get the best model
best_model = random_search.best_estimator_

# Get predicted probabilities
y_pred_proba = best_model.predict_proba(X_test)[:, 1]

# Apply the best threshold
best_threshold = 0.52
y_pred = (y_pred_proba >= best_threshold).astype(int)

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")


# In[ ]:





# In[34]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import fbeta_score, make_scorer
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import optuna

data.dtypes
df=pd.DataFrame(data)


target='Attrition'
X= df.drop(columns=[target, 'EmpID'], axis=1)
y= df[target]

from sklearn.preprocessing import OneHotEncoder


# one-hot encoding
encoder=OneHotEncoder(sparse_output=False)
categorical_columns=X.select_dtypes(include=['object']).columns.tolist()
one_hot_encoded=encoder.fit_transform(X[categorical_columns])
one_hot_df=pd.DataFrame(one_hot_encoded, columns=encoder.get_feature_names_out(categorical_columns))

# Concatenate the one-hot encoded dataframe with the original dataframe
X_encoded = pd.concat([X, one_hot_df], axis=1)

# Drop the original categorical columns
# After one-hot encoding and before scaling
X_encoded = pd.concat([X, one_hot_df], axis=1)
X_encoded = X_encoded.drop(categorical_columns, axis=1)

# Combine X and y into a single dataframe
combined_df = pd.concat([X_encoded, y], axis=1)

# Drop NaN values from the combined dataframe
combined_df = combined_df.dropna()

# Split back into X and y
X = combined_df.drop(columns=[target])
y = combined_df[target]

# Continue with your mapping of y
y = y.map({'Yes': 1, 'No': 0})


from sklearn.preprocessing import StandardScaler
# Now proceed with scaling
scaler = StandardScaler()
scaled_X = scaler.fit_transform(X)

# Apply SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Scale features
scaler = StandardScaler()
scaled_X = scaler.fit_transform(X_resampled)

# Split data
X_train, X_test, y_train, y_test = train_test_split(scaled_X, y_resampled, test_size=0.2, random_state=42)

# Define Optuna objective
def objective(trial):
    params = {
        'max_depth': trial.suggest_int('max_depth', 1, 10),
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.5, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 50, 1000),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1, 10)
    }
    
    model = xgb.XGBClassifier(**params, random_state=42)
    
    # Use cross-validation score
    score = cross_val_score(model, X_train, y_train, cv=5, scoring=make_scorer(fbeta_score, beta=2))
    return score.mean()

# Create and run Optuna study
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

# Get best parameters and train final model
best_params = study.best_params
best_model = xgb.XGBClassifier(**best_params, random_state=42)
best_model.fit(X_train, y_train)

from sklearn.metrics import roc_curve

# Get predicted probabilities
y_pred_proba = best_model.predict_proba(X_test)[:, 1]

# Calculate ROC curve and find optimal threshold
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
J = tpr - fpr
optimal_idx = np.argmax(J)
optimal_threshold = thresholds[optimal_idx]

print(f"Optimal threshold: {optimal_threshold}")

# Make predictions using the optimal threshold
y_pred = (y_pred_proba >= optimal_threshold).astype(int)

# Evaluate the model
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))



# In[ ]:





# In[ ]:





# In[36]:


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc

# Feature Importance Plot
plt.figure(figsize=(12, 8))
feature_importance = best_model.feature_importances_
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5

plt.barh(pos, feature_importance[sorted_idx], align='center')
plt.yticks(pos, X.columns[sorted_idx])
plt.xlabel('Feature Importance')
plt.title('Feature Importance in XGBoost Model')
plt.tight_layout()
plt.show()

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")

# Plot the optimal threshold point
optimal_idx = np.argmax(tpr - fpr)
plt.plot(fpr[optimal_idx], tpr[optimal_idx], 'ro', markersize=8, 
         label=f'Optimal threshold: {thresholds[optimal_idx]:.2f}')
plt.legend()

plt.show()

# Print top 10 important features
feature_importance_df = pd.DataFrame({
    'feature': X.columns,
    'importance': best_model.feature_importances_
}).sort_values('importance', ascending=False)

print("Top 10 Important Features:")
print(feature_importance_df.head(10))


# In[ ]:





# In[ ]:





# In[40]:


# Create a results dataframe
results_df = pd.DataFrame({
    'Actual_Attrition': y_test,
    'Attrition_Probability': y_pred_proba,
    'Predicted_Attrition': y_pred
})

# Show top 10 at-risk employees
print("\nTop 10 employees at risk of attrition:")
at_risk = results_df.sort_values('Attrition_Probability', ascending=False).head(10)
print(at_risk)

# Calculate summary statistics
print("\nSummary:")
print(f"Total employees: {len(results_df)}")
print(f"Predicted attrition: {results_df['Predicted_Attrition'].sum()} ({results_df['Predicted_Attrition'].mean():.2%})")


# In[ ]:





# In[ ]:





# In[41]:


# Get feature names
feature_names = X.columns.tolist()

print("Features used for prediction:")
for i, feature in enumerate(feature_names, 1):
    print(f"{i}. {feature}")

# Save feature names for later use
import joblib
joblib.dump(feature_names, 'feature_names.joblib')


# In[42]:


def manual_prediction(model, scaler, feature_names, threshold):
    # Create a dictionary to store user inputs
    user_input = {}
    
    # Ask for input for each feature
    for feature in feature_names:
        value = input(f"Enter value for {feature}: ")
        # Convert to float if possible, otherwise keep as string
        try:
            user_input[feature] = float(value)
        except ValueError:
            user_input[feature] = value
    
    # Convert user input to DataFrame
    input_df = pd.DataFrame([user_input])
    
    # Ensure all columns from original feature set are present
    for col in feature_names:
        if col not in input_df.columns:
            input_df[col] = 0  # or another appropriate default value
    
    # Reorder columns to match original feature order
    input_df = input_df[feature_names]
    
    # Scale the input
    scaled_input = scaler.transform(input_df)
    
    # Make prediction
    probability = model.predict_proba(scaled_input)[0, 1]
    prediction = 1 if probability >= threshold else 0
    
    return probability, prediction

# Example usage:
# probability, prediction = manual_prediction(best_model, scaler, feature_names, optimal_threshold)
# print(f"Attrition Probability: {probability:.4f}")
# print(f"Prediction: {'Attrition' if prediction == 1 else 'No Attrition'}")


# In[43]:


import joblib

joblib.dump(best_model, 'best_model.joblib')
joblib.dump(scaler, 'scaler.joblib')
joblib.dump(optimal_threshold, 'optimal_threshold.joblib')


# In[46]:


import pandas as pd
import numpy as np
import joblib

# Load saved components
loaded_model = joblib.load('best_model.joblib')
loaded_scaler = joblib.load('scaler.joblib')
loaded_threshold = joblib.load('optimal_threshold.joblib')
feature_names = joblib.load('feature_names.joblib')  # Load the feature names

# Define core features and their types (as before)
core_features = {
    'Age': 'numeric',
    'DailyRate': 'numeric',
    'DistanceFromHome': 'numeric',
    'Education': 'numeric',
    'EnvironmentSatisfaction': 'numeric',
    'JobInvolvement': 'numeric',
    'JobLevel': 'numeric',
    'JobSatisfaction': 'numeric',
    'MonthlyIncome': 'numeric',
    'NumCompaniesWorked': 'numeric',
    'StockOptionLevel': 'numeric',
    'TotalWorkingYears': 'numeric',
    'YearsAtCompany': 'numeric',
    'YearsInCurrentRole': 'numeric',
    'YearsSinceLastPromotion': 'numeric',
    'YearsWithCurrManager': 'numeric',
    'BusinessTravel': 'categorical',
    'Department': 'categorical',
    'EducationField': 'categorical',
    'Gender': 'categorical',
    'JobRole': 'categorical',
    'MaritalStatus': 'categorical',
    'OverTime': 'categorical'
}

def get_user_input(features):
    user_input = {}
    for feature, feature_type in features.items():
        if feature_type == 'numeric':
            value = float(input(f"Enter value for {feature}: "))
        else:
            value = input(f"Enter value for {feature}: ")
        user_input[feature] = value
    return user_input

def encode_categorical(input_data):
    return pd.get_dummies(pd.DataFrame([input_data]), columns=[k for k, v in core_features.items() if v == 'categorical'])

def manual_prediction(model, scaler, threshold, feature_names):
    user_input = get_user_input(core_features)
    input_df = encode_categorical(user_input)
    
    # Ensure all necessary columns are present
    for col in feature_names:
        if col not in input_df.columns:
            input_df[col] = 0
    
    # Reorder columns to match the model's expected input
    input_df = input_df[feature_names]
    
    # Scale the input
    scaled_input = scaler.transform(input_df)
    
    # Make prediction
    probability = model.predict_proba(scaled_input)[0, 1]
    prediction = 1 if probability >= threshold else 0
    
    return probability, prediction

# Make a prediction
probability, prediction = manual_prediction(loaded_model, loaded_scaler, loaded_threshold, feature_names)
print(f"Attrition Probability: {probability:.4f}")
print(f"Prediction: {'Attrition' if prediction == 1 else 'No Attrition'}")

# After training the model
joblib.dump(X.columns.tolist(), 'feature_names.joblib')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




