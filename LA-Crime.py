#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Task 1

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import HeatMap

# Load the data
df = pd.read_csv('Crime_Data.csv')

# Convert date columns to datetime
df['Date Rptd'] = pd.to_datetime(df['Date Rptd'], format='%m/%d/%Y %I:%M:%S %p')
df['DATE OCC'] = pd.to_datetime(df['DATE OCC'], format='%m/%d/%Y %I:%M:%S %p')

# Extract month and year for seasonal analysis
df['Month'] = df['DATE OCC'].dt.month
df['Year'] = df['DATE OCC'].dt.year

# Replace month numbers with month names
month_names = {
    1: 'January', 2: 'February', 3: 'March', 4: 'April', 5: 'May', 6: 'June',
    7: 'July', 8: 'August', 9: 'September', 10: 'October', 11: 'November', 12: 'December'
}
df['Month Name'] = df['Month'].map(month_names)


# In[3]:


# Seasonal patterns: Crime count by month (with month names and gradient colors)
# Calculate crime counts per month
monthly_crime_counts = df['Month Name'].value_counts().reindex(list(month_names.values()))

# Create a color gradient based on crime counts
colors = plt.cm.coolwarm(monthly_crime_counts / monthly_crime_counts.max())  # Normalize counts for color mapping

# Spatial patterns: Crime count by area
plt.figure(figsize=(12, 6))
sns.countplot(x='AREA NAME', data=df, palette='RdBu', order=df['AREA NAME'].value_counts().index)
plt.title('Crime Count by Area')
plt.xlabel('Area')
plt.ylabel('Number of Crimes')
plt.xticks(rotation=90)
plt.show()

# Plot
plt.figure(figsize=(12, 6))
sns.barplot(x=monthly_crime_counts.index, y=monthly_crime_counts.values, palette=colors)
plt.title('Crime Count by Month')
plt.xlabel('Month')
plt.ylabel('Number of Crimes')
plt.xticks(rotation=45)
plt.show()


# In[4]:


# Group crimes into general crime types
def categorize_crime(crime_desc):
    crime_desc_upper = crime_desc.upper()
    
    if crime_desc_upper in [
        'ASSAULT WITH DEADLY WEAPON ON POLICE OFFICER', 'ASSAULT WITH DEADLY WEAPON, AGGRAVATED ASSAULT',
        'ATTEMPTED ROBBERY', 'BATTERY - SIMPLE ASSAULT', 'BATTERY ON A FIREFIGHTER', 
        'BATTERY POLICE (SIMPLE)', 'BATTERY WITH SEXUAL CONTACT', 'CHILD ABUSE (PHYSICAL) - AGGRAVATED ASSAULT',
        'CHILD ABUSE (PHYSICAL) - SIMPLE ASSAULT', 'CRIMINAL HOMICIDE', 
        'INTIMATE PARTNER - AGGRAVATED ASSAULT', 'INTIMATE PARTNER - SIMPLE ASSAULT', 
        'MANSLAUGHTER, NEGLIGENT', 'OTHER ASSAULT', 'RAPE, ATTEMPTED', 'RAPE, FORCIBLE', 'ROBBERY',
        'KIDNAPPING', 'KIDNAPPING - GRAND ATTEMPT'
    ]:
        return 'Violent Crime'
    
    elif crime_desc_upper in [
        'BURGLARY', 'BURGLARY FROM VEHICLE', 'BURGLARY FROM VEHICLE, ATTEMPTED', 'BURGLARY, ATTEMPTED',
        'THEFT', 'THEFT ($950.01 & OVER)', 'VEHICLE - STOLEN', 'BIKE - STOLEN', 'ARSON', 'VANDALISM',
        'BIKE - ATTEMPTED STOLEN', 'BOAT - STOLEN', 'DEFRAUDING INNKEEPER/THEFT OF SERVICES, $950 & UNDER',
        'DEFRAUDING INNKEEPER/THEFT OF SERVICES, OVER $950.01', 'DISHONEST EMPLOYEE - GRAND THEFT',
        'DISHONEST EMPLOYEE - PETTY THEFT', 'DISHONEST EMPLOYEE ATTEMPTED THEFT', 'SHOPLIFTING - ATTEMPT',
        'SHOPLIFTING - PETTY THEFT ($950 & UNDER)', 'SHOPLIFTING-GRAND THEFT ($950.01 & OVER)',
        'THEFT FROM MOTOR VEHICLE - ATTEMPT', 'THEFT FROM MOTOR VEHICLE - GRAND ($950.01 AND OVER)',
        'THEFT FROM MOTOR VEHICLE - PETTY ($950 & UNDER)', 'THEFT FROM PERSON - ATTEMPT', 
        'THEFT PLAIN - ATTEMPT', 'THEFT PLAIN - PETTY ($950 & UNDER)', 'VANDALISM - FELONY ($400 & OVER, ALL CHURCH VANDALISMS)',
        'VANDALISM - MISDEAMEANOR ($399 OR UNDER)', 'VEHICLE - ATTEMPT STOLEN'
    ]:
        return 'Property Crime'
    
    elif crime_desc_upper in [
        'BRIBERY', 'COUNTERFEIT', 'CREDIT CARDS, FRAUD USE ($950 & UNDER', 
        'CREDIT CARDS, FRAUD USE ($950.01 & OVER)', 'DOCUMENT FORGERY / STOLEN FELONY',
        'DOCUMENT WORTHLESS ($200 & UNDER)', 'DOCUMENT WORTHLESS ($200.01 & OVER)',
        'EMBEZZLEMENT, GRAND THEFT ($950.01 & OVER)', 'EMBEZZLEMENT, PETTY THEFT ($950 & UNDER)',
        'EXTORTION', 'UNAUTHORIZED COMPUTER ACCESS', 'THEFT OF IDENTITY', 'FRAUD', 'FORGERY'
    ]:
        return 'White Collar Crime'
    
    elif crime_desc_upper in [
        'DRUGS, TO A MINOR', 'DRUGS', 'WEAPONS POSSESSION/BOMBING', 'WEAPONS',
        'DISORDERLY CONDUCT', 'PIMPING', 'PANDERING', 'HUMAN TRAFFICKING - COMMERCIAL SEX ACTS',
        'HUMAN TRAFFICKING - INVOLUNTARY SERVITUDE', 'INDECENT EXPOSURE', 'PEEPING TOM',
        'PROWLER', 'STALKING', 'VIOLATION OF COURT ORDER', 'VIOLATION OF RESTRAINING ORDER',
        'VIOLATION OF TEMPORARY RESTRAINING ORDER', 'MISCELLANEOUS CRIME'
    ]:
        return 'Other Crime'
    
    else:
        return 'Uncategorized'

# Applying this function to the DataFrame column
df['Crime Category'] = df['Crm Cd Desc'].apply(categorize_crime)

# Spatial patterns: Crime types by area (grouped into general categories with gradient colors)
# Calculate crime counts per area and crime category
area_crime_counts = df.groupby(['AREA NAME', 'Crime Category']).size().unstack(fill_value=0)

# Normalize counts for color mapping
normalized_counts = area_crime_counts.div(area_crime_counts.sum(axis=1), axis=0)

# Plot
plt.figure(figsize=(12, 6))
sns.heatmap(normalized_counts, cmap='coolwarm', annot=True, fmt='.2f', cbar_kws={'label': 'Proportion of Crimes'})
plt.title('Proportion of Crime Types by Area')
plt.xlabel('Crime Category')
plt.ylabel('Area')
plt.xticks(rotation=45)
plt.show()


# In[5]:


# Heat scatter plot of crimes superimposed over a map of Los Angeles
# Filter out rows with missing latitude/longitude values
df = df.dropna(subset=['LAT', 'LON'])

# Create a base map of Los Angeles
la_map = folium.Map(location=[34.0522, -118.2437], zoom_start=11)

# Add a heatmap layer
heat_data = [[row['LAT'], row['LON']] for index, row in df.iterrows()]
HeatMap(heat_data, radius=10).add_to(la_map)

# Save the map to an HTML file
la_map.save('la_crime_heatmap.html')

# Display the map in the notebook
la_map


# In[ ]:


# Task 2

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score

# Prepare the data
X = df[['AREA', 'Vict Age', 'Vict Sex', 'Vict Descent', 'Premis Cd', 'Weapon Used Cd']]
X = pd.get_dummies(X, columns=['Vict Sex', 'Vict Descent', 'Premis Cd', 'Weapon Used Cd'], drop_first=True)
y = df['Crime Category']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Simple Decision Tree Model
dt = DecisionTreeClassifier(random_state=42)  # Initialize Decision Tree
dt.fit(X_train, y_train)  # Train the model
y_pred = dt.predict(X_test)  # Make predictions
print(f'Simple Decision Tree Accuracy: {accuracy_score(y_test, y_pred)}')  # Evaluate accuracy

# Plot the tree
plt.figure(figsize=(20,10))  # Set figure size
plot_tree(dt, filled=True, feature_names=X.columns, class_names=y.unique(), max_depth=3)  # Plot the tree with depth limit
plt.title("Simple Decision Tree")  # Add title
plt.show()

# Pruned Decision Tree (Restricting Depth and Minimum Splits)
dt_pruned = DecisionTreeClassifier(max_depth=10, min_samples_split=10, random_state=42)  # Set pruning parameters
dt_pruned.fit(X_train, y_train)  # Train pruned model
y_pred_pruned = dt_pruned.predict(X_test)  # Make predictions
print(f'Pruned Decision Tree Accuracy: {accuracy_score(y_test, y_pred_pruned)}')  # Evaluate pruned model

# Bagged Decision Tree (Using an ensemble of trees for better performance)
bagged_tree = BaggingClassifier(estimator=DecisionTreeClassifier(), n_estimators=100, random_state=42)
bagged_tree.fit(X_train, y_train)  # Train bagged model
y_pred_bagged = bagged_tree.predict(X_test)  # Make predictions
print(f'Bagged Tree Accuracy: {accuracy_score(y_test, y_pred_bagged)}')  # Evaluate bagged model


# In[ ]:


# Task 3

from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Prepare the data
X = df[['AREA', 'Vict Sex', 'Vict Descent', 'Premis Cd', 'Weapon Used Cd', 'Crm Cd']]
X = pd.get_dummies(X, columns=['Vict Sex', 'Vict Descent', 'Premis Cd', 'Weapon Used Cd', 'Crm Cd'], drop_first=True)
y = df['Vict Age']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# LASSO Model
lasso = Lasso(alpha=0.01)
lasso.fit(X_train, y_train)
y_pred_lasso = lasso.predict(X_test)
print(f'LASSO RMSE: {mean_squared_error(y_test, y_pred_lasso, squared=False)}')

# Ridge Model
ridge = Ridge(alpha=0.01)
ridge.fit(X_train, y_train)
y_pred_ridge = ridge.predict(X_test)
print(f'Ridge RMSE: {mean_squared_error(y_test, y_pred_ridge, squared=False)}')

# Elastic Net Model
elastic_net = ElasticNet(alpha=0.01, l1_ratio=0.5)
elastic_net.fit(X_train, y_train)
y_pred_elastic_net = elastic_net.predict(X_test)
print(f'Elastic Net RMSE: {mean_squared_error(y_test, y_pred_elastic_net, squared=False)}')

# The Ridge model produces the smallest mean squared error, and thus is the most accurate.


# In[ ]:


# Cross validation
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.preprocessing import StandardScaler

# Standardize Features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Define Models
models = {
    "Lasso": Lasso(alpha=0.01),
    "Ridge": Ridge(alpha=0.01),
    "ElasticNet": ElasticNet(alpha=0.01, l1_ratio=0.5)
}

# Custom scorer for RMSE
rmse_scorer = make_scorer(mean_squared_error, squared=False)

# Perform Cross-Validation
for name, model in models.items():
    rmse_scores = cross_val_score(model, X_scaled, y, cv=5, scoring=rmse_scorer)
    print(f"{name} Mean RMSE: {np.mean(rmse_scores):.4f}")

# Run the evaluation function
evaluate_models(models, X_scaled, y, cv=5)


# In[ ]:




