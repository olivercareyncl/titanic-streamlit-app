import streamlit as st
import pandas as pd

# Function to load the dataset
@st.cache_data
def load_data():
    train_df = pd.read_csv('train.csv')
    return train_df

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

if __name__ == "__main__":
    main()
