# app.py
import streamlit as st
from data_overview import data_overview
from survival_analysis import survival_analysis
from predicting_survival import predicting_survival, load_data

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

