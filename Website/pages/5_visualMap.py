import streamlit as st
import pandas as pd
import numpy as np
from oocsi_source import OOCSI
from uuid import uuid4
from streamlit_extras.switch_page_button import switch_page
import random
import shap
import xgboost as xgb
import matplotlib.pyplot as plt

# st.markdown("""<style> 
# .stSlider {
#     padding-bottom: 20px;    
#     }
#     </style> """, 
#     unsafe_allow_html=True)

#Delete this page from the array of pages to visit, this way it cannot be visited twice
if 'profile4' not in st.session_state:
    st.session_state.profiles.remove("visualMap")
    st.session_state.profile4= 'deleted'
    if (len(st.session_state.profiles)>0):
        st.session_state.nextPage4 = random.randint(0, len(st.session_state.profiles)-1)
        st.session_state.lastQuestion= 'no'
    else:
        st.session_state.lastQuestion= 'yes'

if 'profileIndex' not in st.session_state:
    st.session_state.profileIndex= 0    
    

header = st.container()
characteristics = st.container()
prediction = st.container()
explanation = st.container()
footer =st.container()
evaluation = st.container()



nameArray =st.session_state.X_test_names.loc[st.session_state.profileIndex, "Name"].split(',')
name= nameArray[1]+" "+ nameArray[0]

@st.cache_resource
def trainModel(X_train,Y_train):
    model = xgb.XGBClassifier().fit(X_train, Y_train)
    return model


@st.cache_resource
def getSHAPvalues(_model,X_train, Y_train, X_test):
    # compute SHAP values
    explainer = shap.Explainer(_model, X_test)
    shap_values = explainer(X_test)
    return shap_values




def shapPlot(X_test, _shap_values):
    return shap.plots.waterfall(shap_values[st.session_state.profileIndex])


with header:
    st.header(name, anchor='top')
    # st.write("For debugging:")
    # st.write(st.session_state.participantID)
    XGBmodel= trainModel(st.session_state.X_train, st.session_state.Y_train)
    
with characteristics:
    # initialize list of lists
    data = st.session_state.X_test.iloc[st.session_state.profileIndex].values.reshape(1, -1)
    # Create the pandas DataFrame
    df = pd.DataFrame(data, columns=st.session_state.X_test.columns)
    st.dataframe(df)



with prediction:
    # st.header("Prediction")
    prediction =  XGBmodel.predict(st.session_state.X_test.iloc[st.session_state.profileIndex].values.reshape(1, -1))
    probability = XGBmodel.predict_proba(st.session_state.X_test.iloc[st.session_state.profileIndex].values.reshape(1, -1))
    if prediction == 0:
        prob = round((probability[0][0]*100),2)
        st.markdown("There is {}% change that {}  will :red[**not survive**]".format(prob, name) )
    else:
        prob = round((probability[0][1]*100),2)
        st.markdown("There is {}% change that {}  will :green[**survive**]".format(prob, name) )

with explanation:
    st.subheader("Explanation")
    st.markdown("visualmap, more text here")
   

with footer:
    if st.button("New profile"):
        st.session_state.profileIndex= random.randint(0,400)
        st.text("Scroll up to see the new profile")
    # st.markdown("[New profile](#top)")




with evaluation:
    # finished4 = st.checkbox('I am finished looking at the explanation, continue to the questions')
    # if finished4:
    with st.form("my_form4", clear_on_submit=True):
        st.subheader("Evaluation")
        st.write("These questions only ask for your opinion about this specific explanation")
        q1 = st.select_slider(
        '**1**- From the explanation, I **understand** how the algorithm works:',
        options=['totally disagree', 'disagree', 'neutral' , 'agree', 'totally agree'])

        q2 = st.select_slider(
        '**2**- This explanation of how the algorithm works is **satisfying**:',
        options=['totally disagree', 'disagree', 'neutral' , 'agree', 'totally agree'])

        q3 = st.select_slider(
        '**3**- This explanation of how the algorithm works has **sufficient detail**:',
        options=['totally disagree', 'disagree', 'neutral' , 'agree', 'totally agree'])

        q4 = st.select_slider(
        '**4**- This explanation of how the algorithm works seems **complete**:',
        options=['totally disagree', 'disagree', 'neutral' , 'agree', 'totally agree'])

        q5 = st.select_slider(
        '**5**- This explanation of how the algorithm works **tells me how to use it**:',
        options=['totally disagree', 'disagree', 'neutral' , 'agree', 'totally agree'])

        q6 = st.select_slider(
        '**6**- This explanation of how the algorithm works is **useful to my goals**:',
        options=['totally disagree', 'disagree', 'neutral' , 'agree', 'totally agree'])

        q7 = st.select_slider(
        '**7**- This explanation of the algorithm shows me how **accurate** the algorithm is:',
        options=['totally disagree', 'disagree', 'neutral' , 'agree', 'totally agree'])

        q8 = st.select_slider(
        '**8**- This explanation lets me judge when I should **trust and not trust** the algorithm:',
        options=['totally disagree', 'disagree', 'neutral' , 'agree', 'totally agree'])

        # Every form must have a submit button.
        submitted = st.form_submit_button("Submit")
        if submitted:
            # st.write("question 1", q1)
            st.session_state.oocsi.send('EngD_HAII', {
                'participant_ID': st.session_state.participantID,
                'type of explanation': 'counterfactual',
                'q1': q1,
                'q2': q2,
                'q3': q3,
                'q4': q4,
                'q5': q5,
                'q6': q6,
                'q7': q7,
                'q8': q8,
                
                })
            if (st.session_state.lastQuestion =='yes'): 
                switch_page('finalPage')
            else: 
                    switch_page(st.session_state.profiles[st.session_state.nextPage4])