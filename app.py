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
    
    # Section 1: Dataset Overview
    st.subheader("Dataset Overview")
    st.write("""
        This dataset contains information about passengers aboard the Titanic, including 
        features such as their survival status, age, class, fare, and more. Understanding 
        this data is crucial before proceeding to deeper analyses or predictive modeling.
    """)
    
    st.write(f"Number of Rows: {df.shape[0]}")
    st.write(f"Number of Columns: {df.shape[1]}")
    st.write(f"Data Types:\n{df.dtypes}")

    # Section 2: Data Quality Overview (Missing Values and Duplicates)
    st.subheader("Data Quality Overview")
    missing_data = df.isnull().sum() / len(df) * 100
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

    # Section 5: Relationships between Features
    st.subheader("Relationships Between Features")
    st.write("""
        Understanding how different features interact with each other is crucial in determining 
        which features are likely to influence survival predictions.
    """)
    
    # Scatter plot for 'Age' vs 'Fare'
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(x='Age', y='Fare', data=df, ax=ax, color='purple')
    ax.set_title("Age vs Fare")
    st.pyplot(fig)

    # Section 6: Categorical Features Summary
    st.subheader("Categorical Features Overview")
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

