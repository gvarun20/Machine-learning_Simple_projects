import numpy as np
import pickle
import streamlit as st

st.set_page_config(
    page_title="Diabetes Prognosis System",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
    initial_sidebar_state="expanded",
)

#Load the Saved Model
classifier = pickle.load(open('D:\\Projects\\Github\\Diabetes Prognosis System\\classifier_model.sav', 'rb'))

#Function for Prediction
def diabetes(predictive_input):
    predictive_input_array = np.asarray(predictive_input).reshape(1, -1)
    prediction = classifier.predict(predictive_input_array)
    print('Prediction: ',prediction)

    if (prediction[0] == 1):
        return 'Diabetic'
    else:
        return 'Non-Diabetic'

#Main Function
def main():
    #Title
    st.title('Diabetes Prognosis System')

    #Input Features
    # bp = 'High BP', cholesterol = 'High Cholesterol', bmi = 'BMI', stroke = 'Stroke', 
    # heart = 'Heart Disease', alcohol = 'Alcohol Consumption',
    # gen = 'General Health', men = 'Mental Health', phy = 'Physical Health',
    # diffw = 'Difficulty Walking', age = 'Age'

    bp = st.text_input('High BP')
    cholesterol = st.text_input('High Cholesterol')
    bmi = st.text_input('BMI')
    stroke = st.text_input('Stroke') 
    heart = st.text_input('Heart Disease')
    alcohol = st.text_input('Alcohol Consumption')
    gen = st.text_input('General Health')
    men = st.text_input('Mental Health')
    phy = st.text_input('Physical Health')
    diffw = st.text_input('Difficulty Walking')
    age = st.text_input('Age')

    #Prediction
    prediction = ''

    #Prediction Button
    if st.button('Diabetes Result'):
        prediction = diabetes([bp, cholesterol, bmi, stroke, heart, alcohol, gen, men, phy, 
                               diffw, age])
    
    st.success(prediction)


if __name__ == '__main__':
    main()