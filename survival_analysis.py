# survival_analysis.py
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

def survival_analysis(df):
    st.header("Survival Analysis")
    
    st.write("""
        In this section, we will investigate how different features or combinations of features affect the survival rate of passengers. 
        The goal is to understand which factors had the most influence on whether a passenger survived or not.
    """)

    # Exclude 'PassengerId', 'Survived', 'Name', 'Ticket', 'Cabin' from the dropdown options
    available_columns = [col for col in df.columns if col not in ['PassengerId', 'Survived', 'Name', 'Ticket', 'Cabin', 'Age', 'Fare']]

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
        # Cross-tabulation for combinations of features (counts)
        contingency_table = pd.crosstab(df[feature1], df[feature2], df['Survived'], aggfunc='mean').fillna(0)
        count_table = pd.crosstab(df[feature1], df[feature2])  # Counts of passengers for each combination

        st.write("Passenger Count for Each Combination:")
        st.write(count_table)  # Show count of passengers for each combination

        # Visualize survival rate by combination
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(contingency_table, annot=True, cmap="coolwarm", fmt='.2f', ax=ax)
        ax.set_title(f'Survival Rate by {feature1} and {feature2}')
        st.pyplot(fig)
