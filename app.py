import streamlit as st
import pandas as pd
import numpy as np   
import pickle

#importing model
def load_model(file):
    with open(file, 'rb') as f:
        model = pickle.load(f)
    return model

#creating streamlit app
def main():

    #make the app look nicer
    st.set_page_config(page_title='student mark prediction')

    hide_menu = """
                <style>
                
                footer {visibility: hidden; } 
                </style>
    """
    st.markdown(hide_menu, unsafe_allow_html=True)
    st.title('Student mark prediciton')
    
    st.write('')
    st.write('')

    st.markdown("""
    ## What is this?

    This is a simple linear regression model that will predict the marks of a student based on the amount of hours studied.
    This will take one input, the amount of hours studied. The model was trained on a very small set of data and the aim of this project was to better understand the concept of linear regression and get 
    hands on experience with making an end to end app with streamlit as the front end.
    
    **[source code for project](https://github.com/ElPatatone/student-mark-prediction)**
    """)
    st.write('')
    st.write('')

    st.subheader('Make a prediction')
    hours = st.number_input("Hours studied", 0.00, step=0.5)

    if st.button('Predict'):
        hours_reshaped = np.array(hours).reshape(-1,1)
        # st.write(hours_reshaped)

        model = load_model('marks.pkl')
        predicted_marks = model.predict(hours_reshaped)
        st.info("Predicted mark for {} hours is: {}".format(hours, round(predicted_marks[0], 1)))


if __name__ == '__main__':
    main()
