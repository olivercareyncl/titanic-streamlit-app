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

# Function for plotting visualizations with legends
def plot_visualization(df, plot_type):
    fig, ax = plt.subplots()
    if plot_type == "Survival Rate by Gender":
        sns.barplot(x='Sex', y='Survived', data=df, ax=ax)
        ax.set_title('Survival Rate by Gender')
        ax.bar_label(ax.containers[0], fmt='%.2f', color='white')  # Add percentage labels
    elif plot_type == "Survival Rate by Pclass":
        sns.barplot(x='Pclass', y='Survived', data=df, ax=ax)
        ax.set_title('Survival Rate by Passenger Class')
        ax.bar_label(ax.containers[0], fmt='%.2f', color='white')  # Add percentage labels
    elif plot_type == "Age Distribution":
        sns.histplot(df['Age'], kde=True, ax=ax)
        ax.set_title('Age Distribution of Passengers')
    elif plot_type == "Fare Distribution":
        sns.histplot(df['Fare'], kde=True, ax=ax)
        ax.set_title('Fare Distribution of Passengers')
    elif plot_type == "Survival Rate by Age Group":
        bins = [0, 12, 18, 35, 60, 100]
        labels = ['Child', 'Teen', 'Adult', 'Senior', 'Elderly']
        df['AgeGroup'] = pd.cut(df['Age'], bins=bins, labels=labels)
        sns.barplot(x='AgeGroup', y='Survived', data=df, ax=ax)
        ax.set_title('Survival Rate by Age Group')
    elif plot_type == "Survival Rate by Embarked":
        sns.barplot(x='Embarked', y='Survived', data=df, ax=ax)
        ax.set_title('Survival Rate by Embarked Location')
    elif plot_type == "Fare vs Age":
        sns.scatterplot(x='Fare', y='Age', data=df, ax=ax)
        ax.set_title('Fare vs Age')
    elif plot_type == "Pclass vs Age":
        sns.scatterplot(x='Pclass', y='Age', data=df, ax=ax)
        ax.set_title('Pclass vs Age')
    elif plot_type == "Correlation Heatmap":
        corr = df[['Age', 'Fare', 'Pclass', 'Sex']].corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
        ax.set_title('Correlation Heatmap')
    elif plot_type == "Survival Rate by Pclass and Sex":
        sns.countplot(x='Pclass', hue='Sex', data=df, ax=ax)
        ax.set_title('Survival Rate by Pclass and Sex')

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
        plot_type = st.selectbox("Select Visualization", [
            "Survival Rate by Gender", "Survival Rate by Pclass", "Age Distribution", "Fare Distribution",
            "Survival Rate by Age Group", "Survival Rate by Embarked", "Fare vs Age", "Pclass vs Age",
            "Correlation Heatmap", "Survival Rate by Pclass and Sex"
        ])

        # Generate the selected plot
        plot_visualization(train_df, plot_type)

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
