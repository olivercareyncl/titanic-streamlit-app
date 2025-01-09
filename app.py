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

    # Column Information (Meta-data)
    st.subheader("Column Information")
    column_info = pd.DataFrame({
        'Data Type': df.dtypes,
        'Missing Values': df.isnull().sum(),
        'Unique Values': df.nunique(),
        'Example Values': [df[col].dropna().unique()[:3] for col in df.columns]
    })
    st.write(column_info)

    # Missing Data Analysis
    st.subheader("Missing Data Analysis")
    missing_data = df.isnull().sum() / len(df) * 100  # Percentage of missing data
    missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
    
    if len(missing_data) > 0:
        st.bar_chart(missing_data)
    else:
        st.write("No missing data")

    # Summary Statistics
    st.subheader("Summary Statistics")
    st.write(df.describe())

    # Interactive Correlation Heatmap
    st.subheader("Correlation Heatmap")
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
    st.pyplot(fig)

    # Boxplot for 'Age' and 'Fare'
    st.subheader("Boxplots for 'Age' and 'Fare'")
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    sns.boxplot(x=df['Age'], ax=ax[0])
    ax[0].set_title('Age Distribution')
    sns.boxplot(x=df['Fare'], ax=ax[1])
    ax[1].set_title('Fare Distribution')
    st.pyplot(fig)

    # Unique Value Count for Categorical Columns
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
