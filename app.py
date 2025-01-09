import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from xgboost import XGBClassifier

# Function to load the dataset
@st.cache_data
def load_data():
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')  # Assuming you have a 'test.csv' file
    return train_df, test_df

# Create Age Groups
def create_age_groups(df):
    bins = [0, 2, 12, 18, 35, 60, 100]
    labels = ['Infant', 'Child', 'Teen', 'Adult', 'Senior', 'Elderly']
    df['Age Group'] = pd.cut(df['Age'], bins=bins, labels=labels, right=False)
    return df

# Create Fare Groups based on the percentiles
def create_fare_groups(df):
    bins = [0, 7.91, 14.45, 31, 512]  # 0th, 25th, 50th, 75th percentiles, max
    labels = ['Low Fare', 'Medium Fare', 'High Fare', 'Very High Fare']
    df['Fare Group'] = pd.cut(df['Fare'], bins=bins, labels=labels, right=False)
    return df

# Enhanced Data Overview
def data_overview(df):
    st.header("Data Overview")
    
    # Section 1: Dataset Overview
    st.subheader("Dataset Overview")
    st.write("""
        This dataset contains information about passengers aboard the Titanic, including 
        features such as their survival status, age, class, fare, and more. Understanding 
        this data is crucial before proceeding to deeper analyses or predictive modeling.
    """)
    
    st.write(f"Number of Rows: {df.shape[0]}")
    st.write(f"Number of Columns: {df.shape[1]}")
    
    # Show data types in table format
    data_types = pd.DataFrame({
        'Column Name': df.columns,
        'Data Type': df.dtypes
    })
    st.write(data_types)

    # Feature Definitions
    feature_definitions = {
        'PassengerId': 'Unique ID assigned to each passenger.',
        'Pclass': 'Passenger class (1 = 1st; 2 = 2nd; 3 = 3rd).',
        'Name': 'Name of the passenger.',
        'Sex': 'Gender of the passenger (male, female).',
        'Age': 'Age of the passenger in years.',
        'SibSp': 'Number of siblings or spouses aboard the Titanic.',
        'Parch': 'Number of parents or children aboard the Titanic.',
        'Ticket': 'Ticket number.',
        'Fare': 'Fare paid by the passenger.',
        'Cabin': 'Cabin number where the passenger stayed.',
        'Embarked': 'Port of embarkation (C = Cherbourg; Q = Queenstown; S = Southampton).',
        'Survived': 'Survival status (0 = No; 1 = Yes).',
        'Age Group': 'Categorized age group (Infant, Child, Teen, Adult, Senior, Elderly).',
        'Fare Group': 'Categorized fare group (Low Fare, Medium Fare, High Fare, Very High Fare).'
    }

    # Section 2: Feature Definitions
    st.subheader("Feature Definitions")
    for column, definition in feature_definitions.items():
        st.write(f"**{column}**: {definition}")

    # Section 3: Data Quality Overview (Missing Values and Duplicates)
    st.subheader("Data Quality Overview")
    missing_data = df.isnull().sum() / len(df) * 100  # Percentage of missing data
    missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
    
    st.write("Missing Data Percentages:")
    st.bar_chart(missing_data)
    
    # Check for duplicates
    duplicates = df.duplicated().sum()
    st.write(f"Number of duplicate rows: {duplicates}")

    # Section 4: Feature Distribution and Data Types
    st.subheader("Feature Distribution")
    st.write("""
        Understanding the distribution of features, especially numerical ones like 'Age' 
        and 'Fare', is important to identify any skewness, outliers, or patterns in the data.
    """)
    
    # Display histograms for numerical features
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    sns.histplot(df['Age'], kde=True, ax=ax[0], color='skyblue')
    ax[0].set_title('Age Distribution')
    sns.histplot(df['Fare'], kde=True, ax=ax[1], color='orange')
    ax[1].set_title('Fare Distribution')
    st.pyplot(fig)

    # Section 5: Statistical Summary
    st.subheader("Summary Statistics")
    st.write(df.describe())

# Predicting Survival
def predicting_survival(train_df, test_df):
    st.header("Predicting Survival")
    
    st.write("""
        In this section, we will allow you to trial different models and tune their hyperparameters
        to predict if a passenger survived or not on the Titanic. 
        You can evaluate the performance of each model using the test data.
    """)

    # Preprocess the data
    df = train_df.copy()
    df = create_age_groups(df)
    df = create_fare_groups(df)

    # Define features and target variable
    X = df.drop(columns=['PassengerId', 'Survived', 'Name', 'Ticket', 'Cabin'])
    y = df['Survived']
    
    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model selection
    model_choice = st.selectbox("Select Model", ["Logistic Regression", "Random Forest", "XGBoost"])

    # Hyperparameter tuning for each model
    if model_choice == "Logistic Regression":
        C = st.slider("Regularization Strength (C)", 0.01, 10.0, 1.0)
        model = LogisticRegression(C=C, max_iter=1000)
        
    elif model_choice == "Random Forest":
        n_estimators = st.slider("Number of Trees", 10, 200, 100)
        max_depth = st.slider("Max Depth", 2, 20, 10)
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
        
    elif model_choice == "XGBoost":
        n_estimators = st.slider("Number of Trees", 10, 200, 100)
        learning_rate = st.slider("Learning Rate", 0.01, 0.5, 0.1)
        max_depth = st.slider("Max Depth", 2, 20, 10)
        model = XGBClassifier(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth)

    # Train the model
    if st.button("Train Model"):
        model.fit(X_train, y_train)

        # Make predictions on the test data
        y_pred = model.predict(X_test)
        
        # Evaluation metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)

        st.write(f"**Model Performance:**")
        st.write(f"Accuracy: {accuracy:.2f}")
        st.write(f"Precision: {precision:.2f}")
        st.write(f"Recall: {recall:.2f}")
        st.write(f"F1 Score: {f1:.2f}")
        
        # Display confusion matrix
        fig, ax = plt.subplots()
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=["Not Survived", "Survived"], yticklabels=["Not Survived", "Survived"])
        ax.set_title("Confusion Matrix")
        st.pyplot(fig)
        
        # Test data prediction and result visualization
        st.subheader("Predictions on Test Data")
        test_data = test_df.copy()
        test_data = create_age_groups(test_data)
        test_data = create_fare_groups(test_data)
        X_test_data = test_data.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'])
        
        test_predictions = model.predict(X_test_data)
        test_df['Survived Prediction'] = test_predictions
        
        st.write(test_df[['PassengerId', 'Survived Prediction']].head())  # Show the first few predictions

def main():
    st.title("Titanic Survival Prediction App")

    # Sidebar for navigation
    tab = st.sidebar.selectbox("Select Tab", ["Data Overview", "Survival Analysis", "Predicting Survival"])

    # Load dataset
    train_df, test_df = load_data()

    if tab == "Data Overview":
        data_overview(train_df)

    elif tab == "Survival Analysis":
        survival_analysis(train_df)

    elif tab == "Predicting Survival":
        predicting_survival(train_df, test_df)

if __name__ == "__main__":
    main()

