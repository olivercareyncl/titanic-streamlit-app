import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Function to load the dataset
@st.cache_data
def load_data():
    train_df = pd.read_csv('train.csv')
    return train_df

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

    # Section 2: Data Quality Overview (Missing Values and Duplicates)
    st.subheader("Data Quality Overview")
    missing_data = df.isnull().sum() / len(df) * 100  # Percentage of missing data
    missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
    
    st.write("Missing Data Percentages:")
    st.bar_chart(missing_data)
    
    # Check for duplicates
    duplicates = df.duplicated().sum()
    st.write(f"Number of duplicate rows: {duplicates}")

    # Section 3: Feature Distribution and Data Types
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

    # Section 4: Statistical Summary
    st.subheader("Summary Statistics")
    st.write(df.describe())

def survival_analysis(df):
    st.header("Survival Analysis")
    
    st.write("""
        In this section, we will investigate how different features or combinations of features affect the survival rate of passengers. 
        The goal is to understand which factors had the most influence on whether a passenger survived or not.
    """)

    # Exclude 'PassengerId', 'Survived', 'Name', 'Ticket', 'Cabin' from the dropdown options
    available_columns = [col for col in df.columns if col not in ['PassengerId', 'Survived', 'Name', 'Ticket', 'Cabin', 'Age', 'Fare']]

    # Create Age and Fare groups
    df = create_age_groups(df)
    df = create_fare_groups(df)

    # Add the new columns (Age Group and Fare Group) to available columns for the dropdown
    available_columns.extend(['Age Group', 'Fare Group'])

    # Select feature or combination of features to analyze
    feature_column = st.selectbox("Select Feature to Analyze Against Survival", available_columns)

    if feature_column:
        # Plotting survival rate based on selected feature
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.barplot(x=feature_column, y='Survived', data=df, ax=ax, palette="muted")
        ax.set_title(f'Survival Rate by {feature_column}')
        st.pyplot(fig)

    # Investigating combinations of features
    st.subheader("Survival Rate by Feature Combinations")
    
    feature1 = st.selectbox("Select First Feature", available_columns)
    feature2 = st.selectbox("Select Second Feature", available_columns)

    if feature1 and feature2:
        # Cross-tabulation and visualization for combinations of features
        contingency_table = pd.crosstab(df[feature1], df[feature2], df['Survived'], aggfunc='mean').fillna(0)
        st.write(contingency_table)

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(contingency_table, annot=True, cmap="coolwarm", fmt='.2f', ax=ax)
        ax.set_title(f'Survival Rate by {feature1} and {feature2}')
        st.pyplot(fig)

def main():
    st.title("Titanic Survival Prediction App")

    # Sidebar for navigation
    tab = st.sidebar.selectbox("Select Tab", ["Data Overview", "Survival Analysis", "Model Building", "Model Performance"])

    # Load dataset
    train_df = load_data()

    if tab == "Data Overview":
        data_overview(train_df)

    elif tab == "Survival Analysis":
        survival_analysis(train_df)

    elif tab == "Model Building":
        st.header("Model Building and Prediction")
        # Model building code (same as before)
        model_name = st.selectbox("Select Model", ["Logistic Regression", "Random Forest", "XGBoost"])
        model, accuracy, precision, recall, f1, conf_matrix = build_model(train_df, model_name)
        st.write(f"Model Accuracy: {accuracy:.2f}")
        st.write(f"Precision: {precision:.2f}")
        st.write(f"Recall: {recall:.2f}")
        st.write(f"F1 Score: {f1:.2f}")
        fig, ax = plt.subplots()
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=["Not Survived", "Survived"], yticklabels=["Not Survived", "Survived"])
        st.pyplot(fig)

if __name__ == "__main__":
    main()


