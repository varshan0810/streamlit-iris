import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load the Iris dataset
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target, name="species")

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Streamlit app
st.title("Iris Dataset using KNN Classifier")

# Sidebar for user input
st.sidebar.header("User Input Parameters")
def user_input_features():
    sepal_length = st.sidebar.slider("Sepal Length", float(X["sepal length (cm)"].min()), float(X["sepal length (cm)"].max()), float(X["sepal length (cm)"].mean()))
    sepal_width = st.sidebar.slider("Sepal Width", float(X["sepal width (cm)"].min()), float(X["sepal width (cm)"].max()), float(X["sepal width (cm)"].mean()))
    petal_length = st.sidebar.slider("Petal Length", float(X["petal length (cm)"].min()), float(X["petal length (cm)"].max()), float(X["petal length (cm)"].mean()))
    petal_width = st.sidebar.slider("Petal Width", float(X["petal width (cm)"].min()), float(X["petal width (cm)"].max()), float(X["petal width (cm)"].mean()))
    data = {
        "sepal length (cm)": sepal_length,
        "sepal width (cm)": sepal_width,
        "petal length (cm)": petal_length,
        "petal width (cm)": petal_width,
    }
    features = pd.DataFrame(data, index=[0])
    return features

# Get user input
user_input = user_input_features()

# Display user input
st.subheader("User Input Parameters")
st.write(user_input)

# Train the KNN model
k = st.sidebar.slider("Number of Neighbors (k)", 1, 15, 5)
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train)

# Make predictions
prediction = knn.predict(user_input)
prediction_proba = knn.predict_proba(user_input)

# Display prediction
st.subheader("Prediction")
st.write(f"The predicted species is: **{iris.target_names[prediction][0]}**")

# Display prediction probabilities
st.subheader("Prediction Probability")
st.write(prediction_proba)

# Evaluate the model
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
st.subheader("Model Accuracy")
st.write(f"The model's accuracy on the test set is: **{accuracy:.2f}**")