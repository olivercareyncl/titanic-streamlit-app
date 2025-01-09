import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Function to load the dataset
@st.cache_data
def load_data():
    train_df = pd.read_csv('train.csv')
    return train_df

# Enhanced Data Overview
def data_overview(df):
    st.header("Data Overview")
    
    # Dataset preview
    st.subheader("Dataset Preview")
    num_rows = st.slider("Number of rows to display", 5, 100, 10)
    st.write(df.head(num_rows))

    # Missing Data Analysis
    st.subheader("Missing Data Analysis")
    missing_data = df.isnull().sum() / len(df) * 100  # Percentage of missing data
    missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
    
    if len(missing_data) > 0:
        st.bar_chart(missing_data)
    else:
        st.write("No missing data")
    
    # Fill missing values for numerical columns with the median or mean
    df['Age'].fillna(df['Age'].median(), inplace=True)  # Filling missing Age with the median
    df['Fare'].fillna(df['Fare'].median(), inplace=True)  # Filling missing Fare with the median
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)  # Filling missing Embarked with the mode
    df['Cabin'].fillna('Unknown', inplace=True)  # Filling missing Cabin with 'Unknown'

    # Column Information (Meta-data)
    st.subheader("Column Information")
    column_info = pd.DataFrame({
        'Data Type': df.dtypes,
        'Missing Values': df.isnull().sum(),
        'Unique Values': df.nunique(),
        'Example Values': [df[col].dropna().unique()[:3] for col in df.columns]
    })
    st.write(column_info)

    # Summary Statistics
    st.subheader("Summary Statistics")
    st.write(df.describe())

    # Section: Relationships Between Features and Survival
    st.subheader("Relationships Between Features and Survival Rate")

    # For numerical features: Show relationship with Survival using Boxplots
    numerical_features = ['Age', 'Fare']
    for feature in numerical_features:
        st.write(f"**{feature}** vs Survived")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.boxplot(x='Survived', y=feature, data=df, ax=ax, palette='muted')
        ax.set_title(f'{feature} Distribution by Survival')
        st.pyplot(fig)

    # For categorical features: Show relationship with Survival using Bar Plots
    categorical_features = ['Pclass', 'Sex', 'Embarked']
    for feature in categorical_features:
        st.write(f"**{feature}** vs Survived")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.barplot(x=feature, y='Survived', data=df, ax=ax, palette='muted')
        ax.set_title(f'Survival Rate by {feature}')
        st.pyplot(fig)

    # Section: Categorical Columns Overview
    st.subheader("Categorical Columns Overview")
    categorical_columns = df.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        st.write(f"**{col}**")
        st.write(df[col].value_counts())
        st.bar_chart(df[col].value_counts())

def main():
    st.title("Titanic Survival Prediction App")

    # Sidebar for navigation
    tab = st.sidebar.selectbox("Select Tab", ["Data Overview", "Data Visualization", "Model Building", "Model Performance"])

    # Load dataset
    train_df = load_data()

    if tab == "Data Overview":
        data_overview(train_df)

    elif tab == "Data Visualization":
        st.header("Data Visualization")
        plot_type = st.selectbox("Select Visualization", [
            "Survival Rate by Gender", "Survival Rate by Pclass", "Age Distribution", "Fare Distribution",
            "Survival Rate by Age Group"
        ])
        # Generate the selected plot (same as before)
        plot_visualization(train_df, plot_type)

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

