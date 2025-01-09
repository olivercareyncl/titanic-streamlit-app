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

# Load the Titanic dataset
@st.cache_data
def load_data():
    train_df = pd.read_csv('train.csv')
    return train_df

# Function to build the selected model
@st.cache_resource
def build_model(df, model_name):
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    imputer = SimpleImputer(strategy='mean')
    df['Age'] = imputer.fit_transform(df[['Age']])
    df['Fare'] = imputer.fit_transform(df[['Fare']])

    X = df[['Pclass', 'Age', 'Fare', 'Sex']]
    y = df['Survived']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if model_name == 'Logistic Regression':
        model = LogisticRegression(max_iter=500)
    elif model_name == 'Random Forest':
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    elif model_name == 'XGBoost':
        model = xgb.XGBClassifier(eval_metric='logloss', use_label_encoder=False)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)

    return model, accuracy, precision, recall, f1, conf_matrix

# Function for plotting visualizations
def plot_visualization(df, plot_type, x_col, y_col):
    fig, ax = plt.subplots()
    if plot_type == "Bar Plot":
        sns.barplot(x=x_col, y=y_col, data=df, ax=ax)
    elif plot_type == "Scatter Plot":
        sns.scatterplot(x=x_col, y=y_col, data=df, ax=ax)
    elif plot_type == "Line Plot":
        sns.lineplot(x=x_col, y=y_col, data=df, ax=ax)
    st.pyplot(fig)

def main():
    st.title("Titanic Survival Prediction App")

    # Sidebar for navigation
    tab = st.sidebar.selectbox("Select Tab", ["Data Overview", "Data Visualization", "Model Building", "Model Performance"])

    # Load dataset
    train_df = load_data()

    if tab == "Data Overview":
        st.header("Data Overview")
        st.subheader("Data Preview")
        st.write(train_df.head())

        st.subheader("Basic Statistics")
        st.write(train_df.describe())

        st.subheader("Missing Values")
        st.write(train_df.isnull().sum())

    elif tab == "Data Visualization":
        st.header("Data Visualization")

        # Dropdown to select plot type
        plot_type = st.selectbox("Select Plot Type", ["Bar Plot", "Scatter Plot", "Line Plot"])

        # Dropdowns to select x and y variables
        x_variable = st.selectbox("Select X Variable", train_df.columns)
        y_variable = st.selectbox("Select Y Variable", train_df.columns)

        plot_visualization(train_df, plot_type, x_variable, y_variable)

    elif tab == "Model Building":
        st.header("Model Building and Prediction")

        # Model selection dropdown
        model_name = st.selectbox("Select Model", ["Logistic Regression", "Random Forest", "XGBoost"])

        # Build model and display evaluation metrics
        model, accuracy, precision, recall, f1, conf_matrix = build_model(train_df, model_name)

        st.write(f"Model Accuracy: {accuracy:.2f}")
        st.write(f"Precision: {precision:.2f}")
        st.write(f"Recall: {recall:.2f}")
        st.write(f"F1 Score: {f1:.2f}")

        # Display confusion matrix
        fig, ax = plt.subplots()
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=["Not Survived", "Survived"], yticklabels=["Not Survived", "Survived"])
        st.pyplot(fig)

        st.subheader("Predict Survival for a New Passenger")

        # Input fields for user data
        pclass = st.selectbox('Class', [1, 2, 3])
        age = st.slider('Age', 0, 100, 30)
        fare = st.slider('Fare', 0, 500, 50)
        sex = st.selectbox('Sex', ['male', 'female'])

        input_data = pd.DataFrame([[pclass, age, fare, sex]], columns=['Pclass', 'Age', 'Fare', 'Sex'])
        input_data['Sex'] = input_data['Sex'].map({'male': 0, 'female': 1})

        prediction = model.predict(input_data)[0]

        if prediction == 1:
            st.write("Predicted Survival: Survived")
        else:
            st.write("Predicted Survival: Not Survived")

    elif tab == "Model Performance":
        st.header("Model Performance")
        model_name = st.selectbox("Select Model for Evaluation", ["Logistic Regression", "Random Forest", "XGBoost"])

        model, accuracy, precision, recall, f1, conf_matrix = build_model(train_df, model_name)

        st.write(f"Model Accuracy: {accuracy:.2f}")
        st.write(f"Precision: {precision:.2f}")
        st.write(f"Recall: {recall:.2f}")
        st.write(f"F1 Score: {f1:.2f}")

        # Display confusion matrix
        fig, ax = plt.subplots()
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=["Not Survived", "Survived"], yticklabels=["Not Survived", "Survived"])
        st.pyplot(fig)

if __name__ == "__main__":
    main()



