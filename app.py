import pickle
import streamlit as st
import pandas as pd

model=pickle.load(open('Rain-fall-predication-in-Australia-main/Finalmodel.sav','rb'))

def home():
    return 'Welcome'

def Prediction(MinTemp,MaxTemp,Rainfall,Sunshine,Humidity3pm,Cloud9am):
    pred=model.predict([[MinTemp,MaxTemp,Rainfall,Sunshine,Humidity3pm,Cloud9am]])
    if pred==1:
        return "There will be Rainfall Tomorrow "
    else:
        return "There will be No rainFall"

def main():
    st.title('Welcome')
    html_temp = """
        <div style="background-color:tomato;padding:10px">
        <h2 style="color:white;text-align:center;">Rain Fall Prediction ML App </h2>
        </div>
        """
    st.markdown(html_temp,unsafe_allow_html=True)
    MinTemp = st.text_input("Min Temp Today", "num")
    MaxTemp = st.text_input("Max Temp Today", "num")
    Rainfall = st.text_input("RainFall Today", "num")
    Sunshine = st.text_input("Sunshine Today", "num")
    Humidity3pm = st.text_input("Humidity Today", "num")
    Cloud9am = st.text_input("cloud Today", "num")
    result = ""
    if st.button("Predict"):
        result = Prediction(MinTemp,MaxTemp,Rainfall,Sunshine,Humidity3pm,Cloud9am)
    st.success(' {}'.format(result))
    if st.button("About"):
        st.text("Made by")
        st.text("Kashit Duhan")

if __name__ == '__main__':
    main()