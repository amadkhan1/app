import streamlit as st
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
data_url = "https://raw.githubusercontent.com/amadkhan1/app/refs/heads/main/User_Data.csv"
df = pd.read_csv(data_url)
target_column = "class" 
if target_column not in df.columns:
    target_column = df.columns[-1] 
X = df.drop(target_column, axis=1)
y = df[target_column]
non_numeric_columns = X.select_dtypes(exclude=["number"]).columns
st.write("**ASSIGNMENT 04**")
st.write("**Name: Amad Khan  ID: F2021266440**")
if len(non_numeric_columns) > 0:
    st.write(f"Non-numeric columns: {non_numeric_columns}")
    X = X.drop(non_numeric_columns, axis=1)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
model = SVC(kernel='rbf', C=1.0, gamma='scale')
model.fit(X_train, y_train)
st.title("SVM Prediction App")
st.write("This app predicts the class using a pre-trained SVM model.")
st.sidebar.header("User Input Features")
def user_input_features():
    inputs = {}
    for feature in X.columns:
        inputs[feature] = st.sidebar.number_input(f"{feature}", float(df[feature].min()), float(df[feature].max()), float(df[feature].mean()))
    return pd.DataFrame(inputs, index=[0])
input_df = user_input_features()
st.subheader("User Input")
st.write(input_df)
if st.button("Predict"):
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)
    st.subheader("Prediction")
    if prediction[0] == 0:
        st.write("Prediction: You are **not likely to make a purchase**.")
    else:
        st.write("Prediction: You are **likely to make a purchase**.")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    st.subheader("Model Performance on Test Data")
    st.write(f"Accuracy: {accuracy:.2f}")
    st.write(f"Precision: {precision:.2f}")
    st.write(f"Recall: {recall:.2f}")
    st.write(f"F1 Score: {f1:.2f}")
