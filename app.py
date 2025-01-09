import streamlit as st
import pandas as pd

# Function to load the dataset
@st.cache
def load_data():
    train_df = pd.read_csv('train.csv')
    return train_df

# Display Data
def display_data(df):
    st.title("Titanic Dataset")
    st.subheader("Data Overview")
    st.write(df.head())

def main():
    # Load the data
    train_df = load_data()
    
    # Show data
    display_data(train_df)

if __name__ == "__main__":
    main()
