import streamlit as st
import joblib 
import numpy as np 
import pandas as pd

@st.cache_resource
def load_model():
    model = joblib.load("predict_model.pkl")
    return model

@st.cache_data
def load_accuracy():
    with open("results.txt", "r") as f:
        accuracy = f.read().strip()
    return accuracy

def preprocess_input(input_data):

    df = pd.DataFrame([input_data])
    
    categorical_features = [
        "mainroad", "guestroom", "basement", 
        "hotwaterheating", "airconditioning", "prefarea", 
        "furnishingstatus"
    ]
    
    df = pd.get_dummies(df, columns=categorical_features, drop_first=False)
    
    expected_features = ["area","bedrooms","bathrooms","stories","parking","mainroad_No","mainroad_Yes",
        "guestroom_No","guestroom_Yes","basement_No","basement_Yes","hotwaterheating_No","hotwaterheating_Yes",
        "airconditioning_No","airconditioning_Yes","prefarea_No","prefarea_Yes","furnishingstatus_Furnished",
        "furnishingstatus_semi-furnished","furnishingstatus_unfurnished"
    ]
    
    df = df.reindex(columns=expected_features, fill_value=0)
    return df

model = load_model()
accuracy = load_accuracy()

st.markdown("""
<style>
/* Set background color */
body {
    background-color: #F5F5F5;
}
/* Customize title */
.title {
    font-size: 30px;
    color: #117A65;
    text-align: center;
    font-weight: bold;
}
/* Customize subtitle */
.subtitle {
    font-size: 20px;
    color: #117A65;
    text-align: center;
    font-style: italic; 
}
/* Customize sidebar */
.sidebar-title {
    font-size: 24px;
    color: red;
    text-align: center;
    font-weight: bold;    
}
/* Center align main content */
.main-content {
    max-width: 80%;
    margin: auto;
    padding: 20px;
    background-color: #FFFFFF;
    border-radius: 10px;
    box-shadow: 0px 0px 10px rgba(0,0,0,0.1);
}
</style>
""", unsafe_allow_html=True)

st.markdown("<p class='title'>Linear Regression Model for House Price Prediction</p>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Enter details below to get a prediction</p>", unsafe_allow_html=True)

st.sidebar.markdown("<p class='sidebar-title'>Model Performance</p>", unsafe_allow_html=True)
st.sidebar.write(f"### Model Accuracy: **{accuracy}%**")

st.markdown("<div class='main-content'>", unsafe_allow_html=True)
st.write("### Please enter the required details:")

with st.form(key="prediction_form"):

    area = st.number_input("Area (in square feet)", min_value=0.0, step=0.1, format="%.2f")
    bedrooms = st.number_input("Number of Bedrooms", min_value=1, max_value=4, step=1, value=2)
    bathrooms = st.number_input("Number of Bathrooms", min_value=1, max_value=3, step=1, value=1)
    stories = st.number_input("Number of Stories", min_value=1, max_value=4, step=1, value=1)
    
    mainroad = st.selectbox("Main Road", options=["Yes", "No"])
    guestroom = st.selectbox("Guest Room", options=["Yes", "No"])
    basement = st.selectbox("Basement", options=["Yes", "No"])
    hotwaterheating = st.selectbox("Hot Water Heating", options=["Yes", "No"])
    airconditioning = st.selectbox("Air Conditioning", options=["Yes", "No"])
    prefarea = st.selectbox("Preferred Area", options=["Yes", "No"])
    
    parking = st.number_input("Parking (Number of spaces)", min_value=0, max_value=3, step=1, value=1)
    furnishingstatus = st.selectbox("Furnishing Status", options=["Furnished", "semi-furnished", "unfurnished"])
    
    submit_button = st.form_submit_button(label="Predict")

if submit_button:

    input_data = {
        "area": area,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "stories": stories,
        "mainroad": mainroad,
        "guestroom": guestroom,
        "basement": basement,
        "hotwaterheating": hotwaterheating,
        "airconditioning": airconditioning,
        "parking": parking,
        "prefarea": prefarea,
        "furnishingstatus": furnishingstatus
    }
    

    input_df = preprocess_input(input_data)
    
    try:

        prediction = model.predict(input_df)[0]
        st.success(f"### Predicted House Price: ${prediction:,.2f}")
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")

st.markdown("</div>", unsafe_allow_html=True)

st.title("🚀 Streamlit is Running!")
st.write("If you see this message, Streamlit is working correctly.")

if st.button("Click Me"):
    st.write("✅ Button Clicked!")
