import streamlit as st
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load dataset
data_url = "https://raw.githubusercontent.com/amadkhan1/app/refs/heads/main/User_Data.csv"
df = pd.read_csv(data_url)

# Check column names to ensure correct target column is used
st.write("Column Names in Dataset:", df.columns)

# Preprocess data
# Dynamically check for the target column name
target_column = "class"  # Default to 'class', but this can be adjusted
if target_column not in df.columns:
    target_column = df.columns[-1]  # Use the last column as target if 'class' is not present

X = df.drop(target_column, axis=1)
y = df[target_column]

# Check for non-numeric columns and handle them
non_numeric_columns = X.select_dtypes(exclude=["number"]).columns
if len(non_numeric_columns) > 0:
    st.write(f"Non-numeric columns: {non_numeric_columns}")
    # Optionally, you can drop or encode these columns
    X = X.drop(non_numeric_columns, axis=1)  # Dropping non-numeric columns

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train SVM model
model = SVC(kernel='rbf', C=1.0, gamma='scale')
model.fit(X_train, y_train)

# Streamlit app
st.title("SVM Prediction App")
st.write("This app predicts the class using a pre-trained SVM model.")

# User input form
st.sidebar.header("User Input Features")
def user_input_features():
    inputs = {}
    for feature in X.columns:
        inputs[feature] = st.sidebar.number_input(f"{feature}", float(df[feature].min()), float(df[feature].max()), float(df[feature].mean()))
    return pd.DataFrame(inputs, index=[0])

input_df = user_input_features()

# Display user input
st.subheader("User Input")
st.write(input_df)

# Prediction
if st.button("Predict"):
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)

    # Display prediction
    st.subheader("Prediction")
    st.write(f"Predicted class: {prediction[0]}")

    # Evaluate model
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
