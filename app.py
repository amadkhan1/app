code = """
# Copy and paste the Streamlit app code here
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Title of the application
st.title("SVM Prediction App")

@st.cache_data
def train_svm_model():
    df = pd.read_csv('/content/User_Data.csv')  # Replace with your dataset
    X = df.drop(columns=['Purchased'])  # Adjust the target column name
    y = df['Purchased']
    model = SVC(kernel='rbf', gamma='scale', C=1.0)
    model.fit(X, y)
    return model, X.columns

model, feature_names = train_svm_model()

st.header("Enter Data for Prediction")
user_input = {}
for feature in feature_names:
    user_input[feature] = st.number_input(f"Enter {feature}:", value=0.0)

input_data = np.array([list(user_input.values())]).reshape(1, -1)
if st.button("Predict"):
    prediction = model.predict(input_data)
    if prediction[0] == 1:
        st.success("Prediction: Purchased")
    else:
        st.error("Prediction: Not Purchased")

if st.checkbox("Show Model Accuracy"):
    st.write("Training Accuracy:", accuracy_score(y, model.predict(X)))
"""

with open('app.py', 'w') as f:
    f.write(code)
print("Streamlit app file 'app.py' created!")
