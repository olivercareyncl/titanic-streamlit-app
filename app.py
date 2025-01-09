import streamlit as st
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer

# Function to load the dataset (cache data)
@st.cache_data
def load_data():
    train_df = pd.read_csv('train.csv')
    return train_df

# Data Preprocessing and Model Building (cache the model)
@st.cache_resource
def build_model(df):
    # Feature Engineering
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    
    # Handle missing values for 'Age' and 'Fare'
    imputer = SimpleImputer(strategy='mean')
    df['Age'] = imputer.fit_transform(df[['Age']])
    df['Fare'] = imputer.fit_transform(df[['Fare']])
    
    X = df[['Pclass', 'Age', 'Fare', 'Sex']]
    y = df['Survived']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Logistic Regression Model
    model = LogisticRegression(max_iter=500)
    model.fit(X_train, y_train)
    
    # Evaluate Model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return model, accuracy

# Display Data, Stats, Visualization, and Model Accuracy
def display_data(df):
    st.title("Titanic Dataset")
    
    st.subheader("Data Overview")
    st.write(df.head())
    
    st.subheader("Basic Statistics")
    st.write(df.describe())

    st.subheader("Missing Values")
    st.write(df.isnull().sum())

    st.subheader("Survival Rate by Gender")
    sns.barplot(x='Sex', y='Survived', data=df)
    st.pyplot()

    # Build model and show accuracy
    model, accuracy = build_model(df)
    st.write(f"Model Accuracy: {accuracy:.2f}")

def main():
    # Load the data
    train_df = load_data()
    
    # Show data, stats, visualization, and model accuracy
    display_data(train_df)

if __name__ == "__main__":
    main()

