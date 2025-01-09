import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer

# Function to load the dataset
@st.cache_data
def load_data():
    train_df = pd.read_csv('train.csv')
    return train_df

# Function to build the Logistic Regression model
@st.cache_resource
def build_model(df):
    # Feature Engineering: Convert 'Sex' to numeric values
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    
    # Handle missing values using SimpleImputer for 'Age' and 'Fare'
    imputer = SimpleImputer(strategy='mean')
    df['Age'] = imputer.fit_transform(df[['Age']])
    df['Fare'] = imputer.fit_transform(df[['Fare']])
    
    # Prepare the features (X) and target (y)
    X = df[['Pclass', 'Age', 'Fare', 'Sex']]
    y = df['Survived']
    
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Logistic Regression Model
    model = LogisticRegression(max_iter=500)
    model.fit(X_train, y_train)
    
    # Make predictions and evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return model, accuracy

def main():
    # Load the data
    train_df = load_data()
    
    # Display the dataset
    st.title("Titanic Dataset")
    
    st.subheader("Data Overview")
    st.write(train_df.head())
    
    # Display Basic Statistics
    st.subheader("Basic Statistics")
    st.write(train_df.describe())
    
    # Show missing values
    st.subheader("Missing Values")
    st.write(train_df.isnull().sum())
    
    # Display a visualization: Survival Rate by Gender
    st.subheader("Survival Rate by Gender")
    
    # Create the figure
    fig, ax = plt.subplots()
    sns.barplot(x='Sex', y='Survived', data=train_df, ax=ax)
    
    # Pass the figure to st.pyplot()
    st.pyplot(fig)

    # Build the Logistic Regression model and show accuracy
    model, accuracy = build_model(train_df)
    st.write(f"Model Accuracy: {accuracy:.2f}")

if __name__ == "__main__":
    main()

