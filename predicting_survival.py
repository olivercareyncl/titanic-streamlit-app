# predicting_survival.py
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from xgboost import XGBClassifier
from sklearn.impute import SimpleImputer

def create_age_groups(df):
    bins = [0, 2, 12, 18, 35, 60, 100]
    labels = ['Infant', 'Child', 'Teen', 'Adult', 'Senior', 'Elderly']
    df['Age Group'] = pd.cut(df['Age'], bins=bins, labels=labels, right=False)
    return df

def create_fare_groups(df):
    bins = [0, 7.91, 14.45, 31, 512]  # 0th, 25th, 50th, 75th percentiles, max
    labels = ['Low Fare', 'Medium Fare', 'High Fare', 'Very High Fare']
    df['Fare Group'] = pd.cut(df['Fare'], bins=bins, labels=labels, right=False)
    return df

def encode_categorical_columns(df):
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    df = pd.get_dummies(df, columns=['Embarked', 'Age Group', 'Fare Group'], drop_first=True)
    return df

def handle_missing_values(df):
    imputer = SimpleImputer(strategy='mean')
    df[['Age', 'Fare']] = imputer.fit_transform(df[['Age', 'Fare']])
    return df

def predicting_survival(train_df, test_df):
    st.header("Predicting Survival")
    
    st.write("""
        In this section, we will allow you to trial different models and tune their hyperparameters
        to predict if a passenger survived or not on the Titanic. 
        You can evaluate the performance of each model using the test data.
    """)

    # Preprocess the data
    df = train_df.copy()
    df = create_age_groups(df)
    df = create_fare_groups(df)
    df = handle_missing_values(df)
    df = encode_categorical_columns(df)

    # Define features and target variable
    X = df.drop(columns=['PassengerId', 'Survived', 'Name', 'Ticket', 'Cabin'])
    y = df['Survived']
    
    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model selection
    model_choice = st.selectbox("Select Model", ["Random Forest", "XGBoost"])

    # Hyperparameter tuning for each model
    if model_choice == "Random Forest":
        n_estimators = st.slider("Number of Trees", 10, 200, 100)
        max_depth = st.slider("Max Depth", 2, 20, 10)
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
        
    elif model_choice == "XGBoost":
        n_estimators = st.slider("Number of Trees", 10, 200, 100)
        learning_rate = st.slider("Learning Rate", 0.01, 0.5, 0.1)
        max_depth = st.slider("Max Depth", 2, 20, 10)
        model = XGBClassifier(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth)

    # Train the model
    if st.button("Train Model"):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)

        st.write(f"**Model Performance:**")
        st.write(f"Accuracy: {accuracy:.2f}")
        st.write(f"Precision: {precision:.2f}")
        st.write(f"Recall: {recall:.2f}")
        st.write(f"F1 Score: {f1:.2f}")
        
        fig, ax = plt.subplots()
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=["Not Survived", "Survived"], yticklabels=["Not Survived", "Survived"])
        ax.set_title("Confusion Matrix")
        st.pyplot(fig)
        
        # Test data prediction and result visualization
        st.subheader("Predictions on Test Data")
        test_data = test_df.copy()
        test_data = create_age_groups(test_data)
        test_data = create_fare_groups(test_data)
        test_data = handle_missing_values(test_data)
        test_data = encode_categorical_columns(test_data)
        X_test_data = test_data.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'])
        
        test_predictions = model.predict(X_test_data)
        test_df['Survived Prediction'] = test_predictions
        
        st.write(test_df[['PassengerId', 'Survived Prediction']].head())  # Show the first few predictions
