
import streamlit as st
import pandas as pd
import pickle

# Load the trained RandomForest model and scaler
with open('heart_model.pkl', 'rb') as file:
    model = pickle.load(file)
with open('sch.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Define the Streamlit app
st.title('Heart Disease Prediction App')

# Add the description
st.markdown("""
Heart disease is a major cause of death around the world, but finding it early can save lives. This app helps you check your risk based on your health information, so you can take steps to stay healthy. It's easy to use, keeps your data safe, and gives you the information you need to make good choices about your heart health. Before that, please have a look on the symptoms of heart diseases.
""")
# Display the image
st.image("https://images.everydayhealth.com/images/seo-graphic-content-initiative/select-warning-signs-of-heart-disease.png?sfvrsn=32d246f4_3", caption='Symptoms of Heart Disease', use_column_width = True)

# User input for prediction
def user_input_features():
    st.sidebar.markdown("### Fill in the details to assess your risk of heart disease:")
    age = st.sidebar.number_input('Age', min_value=0, max_value=100, value=50)
    sex = st.sidebar.selectbox('Sex', (0, 1))
    cp = st.sidebar.selectbox('Chest Pain Type (cp)', (0, 1, 2, 3))
    trestbps = st.sidebar.number_input('Resting Blood Pressure (trestbps)', min_value=0, max_value=200, value=120)
    chol = st.sidebar.number_input('Serum Cholestoral (chol)', min_value=0, max_value=600, value=200)
    fbs = st.sidebar.selectbox('Fasting Blood Sugar (fbs > 120 mg/dl)', (0, 1))
    restecg = st.sidebar.selectbox('Resting Electrocardiographic Results (restecg)', (0, 1, 2))
    thalach = st.sidebar.number_input('Maximum Heart Rate Achieved (thalach)', min_value=0, max_value=250, value=150)
    exang = st.sidebar.selectbox('Exercise Induced Angina (exang)', (0, 1))
    oldpeak = st.sidebar.number_input('ST Depression Induced by Exercise (oldpeak)', min_value=0.0, max_value=10.0, value=1.0)
    slope = st.sidebar.selectbox('Slope of the Peak Exercise ST Segment (slope)', (0, 1, 2))
    ca = st.sidebar.number_input('Number of Major Vessels (ca)', min_value=0, max_value=4, value=0)
    thal = st.sidebar.selectbox('Thalassemia (thal)', (0, 1, 2, 3))
    
    data = {
        'age': age,
        'sex': sex,
        'cp': cp,
        'trestbps': trestbps,
        'chol': chol,
        'fbs': fbs,
        'restecg': restecg,
        'thalach': thalach,
        'exang': exang,
        'oldpeak': oldpeak,
        'slope': slope,
        'ca': ca,
        'thal': thal
    }
    
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Display user input
st.subheader('User Input Parameters')
st.markdown("If details entered are correct. Press the Predict button")
st.write(input_df)

# Prediction
prediction = model.predict(input_df)
prediction_proba = model.predict_proba(input_df)


if st.button('Predict'):
    input_scaled = scaler.transform(input_df)  # Ensure the input is scaled as per training
    prediction = model.predict(input_scaled)
    prediction_proba = model.predict_proba(input_scaled)
    
    if prediction[0] == 1:
        st.write('<h3 style="color: red;">Heart Disease</h3>', unsafe_allow_html=True)
        st.image("https://previews.123rf.com/images/gloly67/gloly671807/gloly67180700005/104596443-cartoon-character-sadness-heart-with-nameplate-help-unhealthy-heart-concept-icon-design-vector.jpg", caption='You are at Risk', use_column_width = True)
    else:
        st.write('<h3 style="color: green;">No Heart Disease</h3>', unsafe_allow_html=True)
        st.image("https://static8.depositphotos.com/1400067/893/i/450/depositphotos_8936376-stock-photo-heart-mascot.jpg", caption='You are Safe', use_column_width = True)


# Display the prediction probability
st.subheader('Prediction Probability')
st.write(prediction_proba)
