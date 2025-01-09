import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.impute import SimpleImputer
import xgboost as xgb

# Function to load the dataset
@st.cache_data
def load_data():
    train_df = pd.read_csv('train.csv')
    return train_df

# Function to build and return the model based on user's selection
@st.cache_resource
def build_model(df, model_name):
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
    
    # Initialize the model based on user selection
    if model_name == 'Logistic Regression':
        model = LogisticRegression(max_iter=500)
    elif model_name == 'Random Forest':
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    elif model_name == 'XGBoost':
        model = xgb.XGBClassifier(eval_metric='logloss', use_label_encoder=False)
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    return model, accuracy, precision, recall, f1, conf_matrix

# Function to make predictions based on user input
def predict_survival(model):
    # Get user input
    st.subheader("Predict Survival for a New Passenger")
    
    pclass = st.selectbox('Class', [1, 2, 3])
    age = st.slider('Age', 0, 100, 30)
    fare = st.slider('Fare', 0, 500, 50)
    sex = st.selectbox('Sex', ['male', 'female'])
    
    # Prepare input data for prediction
    input_data = pd.DataFrame([[pclass, age, fare, sex]], columns=['Pclass', 'Age', 'Fare', 'Sex'])
    input_data['Sex'] = input_data['Sex'].map({'male': 0, 'female': 1})
    
    # Predict survival
    prediction = model.predict(input_data)[0]
    
    # Display result
    if prediction == 1:
        st.write("Predicted Survival: Survived")
    else:
        st.write("Predicted Survival: Not Survived")

# Function to display the confusion matrix
def plot_confusion_matrix(conf_matrix):
    fig, ax = plt.subplots()
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=["Not Survived", "Survived"], yticklabels=["Not Survived", "Survived"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    st.pyplot(fig)

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
    
    # Model selection
    model_name = st.selectbox("Select Model", ['Logistic Regression', 'Random Forest', 'XGBoost'])
    
    # Build the selected model and show evaluation metrics
    model, accuracy, precision, recall, f1, conf_matrix = build_model(train_df, model_name)
    
    # Display Model Evaluation Metrics
    st.write(f"Model Accuracy: {accuracy:.2f}")
    st.write(f"Model Precision: {precision:.2f}")
    st.write(f"Model Recall: {recall:.2f}")
    st.write(f"Model F1 Score: {f1:.2f}")
    
    # Plot Confusion Matrix
    plot_confusion_matrix(conf_matrix)
    
    # Get prediction for a new passenger
    predict_survival(model)

if __name__ == "__main__":
    main()



