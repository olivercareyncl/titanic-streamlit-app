# data_overview.py
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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
