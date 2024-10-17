import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Set the page layout to wide
st.set_page_config(layout="wide")

# Title
st.title("📊 Customer Churn Prediction")

# Read Data
df = pd.read_csv("data/customer_churn_dataset-testing-master.csv")
st.subheader("Data Preview")
st.dataframe(df)

# Visualizations

# Churn by Subscription Type
st.subheader("Churn by Subscription Type")
churn_subscription = df.groupby(['Subscription Type', 'Churn']).size().unstack().reset_index()
fig_churn_subscription = px.bar(churn_subscription, 
                                 x='Subscription Type', 
                                 y=[0, 1], 
                                 title='Churn by Subscription Type',
                                 labels={'value': 'Count', 'variable': 'Churn'}, 
                                 barmode='group', 
                                 color_discrete_sequence=px.colors.qualitative.Plotly)
st.plotly_chart(fig_churn_subscription)

# Age Distribution
st.subheader("Age Distribution")
age_distribution = df['Age'].value_counts().sort_index().reset_index()
age_distribution.columns = ['Age', 'Count']
fig_age_distribution = px.bar(age_distribution, 
                               x='Age', 
                               y='Count', 
                               title='Age Distribution',
                               labels={'Count': 'Number of Customers'})
st.plotly_chart(fig_age_distribution)

# Churn by Gender
st.subheader("Churn by Gender")
churn_gender = df.groupby(['Gender', 'Churn']).size().unstack().reset_index()
fig_churn_gender = px.bar(churn_gender, 
                           x='Gender', 
                           y=[0, 1], 
                           title='Churn by Gender',
                           labels={'value': 'Count', 'variable': 'Churn'}, 
                           barmode='group', 
                           color_discrete_sequence=px.colors.qualitative.Plotly)
st.plotly_chart(fig_churn_gender)

# Usage Frequency by Gender
st.subheader("Usage Frequency by Gender")
usage_frequency = df[['Gender', 'Usage Frequency']].groupby('Gender').mean().reset_index()
fig_usage_frequency = px.bar(usage_frequency, 
                              x='Gender', 
                              y='Usage Frequency', 
                              title='Average Usage Frequency by Gender',
                              labels={'Usage Frequency': 'Average Usage Frequency'},
                              color='Usage Frequency',
                              color_continuous_scale=px.colors.sequential.Plasma)
st.plotly_chart(fig_usage_frequency)

# Contract Length vs. Payment Delay
st.subheader("Contract Length vs. Payment Delay")
contract_payment = df.groupby('Contract Length')['Payment Delay'].mean().reset_index()
fig_contract_payment = px.bar(contract_payment, 
                               x='Contract Length', 
                               y='Payment Delay', 
                               title='Average Payment Delay by Contract Length',
                               labels={'Payment Delay': 'Average Payment Delay'},
                               color='Payment Delay',
                               color_continuous_scale=px.colors.sequential.Viridis)
st.plotly_chart(fig_contract_payment)

# Prediction Section
st.subheader("🔍 Model Training and Evaluation")

# Load train dataset
train_data = pd.read_csv('data/customer_churn_dataset-training-master.csv')

# Feature Engineering Function
def feature_engineering(data):
    data = pd.get_dummies(data, columns=['Gender', 'Subscription Type', 'Contract Length'], drop_first=True)
    data['Spend_per_Tenure'] = data['Total Spend'] / data['Tenure']
    return data.drop(columns=['CustomerID', 'Last Interaction'])

train_data = feature_engineering(train_data)

# Handle NaN values in target variable
train_data = train_data.dropna(subset=['Churn'])

# Split features and target
X = train_data.drop(columns=['Churn'])
y = train_data['Churn']

# Split into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Handle missing values in training set
imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imputed)
X_test_scaled = scaler.transform(X_test_imputed)

# Initialize Logistic Regression model
log_reg = LogisticRegression(max_iter=1000)  # Increased max_iter for convergence

# Train the model
log_reg.fit(X_train_scaled, y_train)

# Make predictions on the test set
y_pred = log_reg.predict(X_test_scaled)

# Calculate accuracy and display results
accuracy = accuracy_score(y_test, y_pred)
st.write(f"### Model Accuracy: **{accuracy:.2f}**")

# Display confusion matrix and classification report
conf_matrix = confusion_matrix(y_test, y_pred)
st.subheader("Confusion Matrix")

# Create a figure for the confusion matrix plot using Plotly
conf_matrix_df = pd.DataFrame(conf_matrix, index=['Not Churned', 'Churned'], columns=['Not Churned', 'Churned'])
fig_conf_matrix = px.imshow(conf_matrix_df, text_auto=True,
                             labels=dict(x="Predicted", y="Actual", color="Count"),
                             title="Confusion Matrix",
                             color_continuous_scale='Blues')
st.plotly_chart(fig_conf_matrix)

st.subheader("Classification Report")
report = classification_report(y_test, y_pred)
st.text(report)

# Display prediction results for a few records in the test set
results_df = pd.DataFrame({
    'Actual Churn': y_test,
    'Predicted Churn': y_pred
}).reset_index(drop=True)