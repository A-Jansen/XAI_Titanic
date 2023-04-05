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
import streamlit.components.v1 as components


#Delete this page from the array of pages to visit, this way it cannot be visited twice
if 'profile4' not in st.session_state:
    st.session_state.pages.remove("visualMap")
    st.session_state.profile4= 'deleted'
    if (len(st.session_state.pages)>0):
        st.session_state.nextPage4 = random.randint(0, len(st.session_state.pages)-1)
        st.session_state.lastQuestion= 'no'
    else:
        st.session_state.lastQuestion= 'yes'


if 'index4' not in st.session_state:
    st.session_state.index4= 0    

if 'profileIndex' not in st.session_state:
    st.session_state.profileIndex= st.session_state.profileIndices[st.session_state.index4]       
    
    

header1, header2, header3 = st.columns([1,2,1])
characteristics1, characteristics2, characteristics3 = st.columns([1,2,1])
prediction1, prediction2, prediction3 =st.columns([1,2,1])
explanationheader1,explanationheader2, explanationheader3 = st.columns([1,2,1])
explanation1, explanation2, explanation3 = st.columns([1,6,1])
footer1, footer2, footer3 =st.columns([1,2,1])
evaluation1, evaluation2, evaluation3 = st.columns([1,2,1])



name= st.session_state.X_test_names.loc[st.session_state.profileIndex, "Name"]


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


with header2:
    st.header('Visual Method for XAI')
    st.markdown('''In this part, a new method of Explainability was implemented using more visual techniques for communicating of the model
    predictions and the features influence. The values showed when clicked on each feature (title, Age, deck, ...) were obtained using the SHAP algorithm. 
    Let yourself play with it and tell us how easy it was to understand the model prediciton and the influence of the features!

        ''')
    st.markdown("Click on the image to see how each attribute contributed and hover over them to see the SHAP values")
    # st.subheader(name, anchor='top')
    XGBmodel= trainModel(st.session_state.X_train, st.session_state.Y_train)
    
with characteristics2:
    # initialize list of lists
    data = st.session_state.X_test.iloc[st.session_state.profileIndex].values.reshape(1, -1)
    # Create the pandas DataFrame
    df = pd.DataFrame(data, columns=st.session_state.X_test.columns)
    # st.dataframe(df)



with prediction2:
    st.subheader("Prediction")
    prediction =  XGBmodel.predict(st.session_state.X_test.iloc[st.session_state.profileIndex].values.reshape(1, -1))
    probability = XGBmodel.predict_proba(st.session_state.X_test.iloc[st.session_state.profileIndex].values.reshape(1, -1))
    if prediction == 0:
        prob = round((probability[0][0]*100),2)
        st.markdown("The model predicts with {}% probability  that {}  will :red[**not survive**]".format(prob, name) )
    else:
        prob = round((probability[0][1]*100),2)
        st.markdown("The model predicts with {}% probability  that {}  will :green[**survive**]".format(prob, name) )
    
# with explanationheader2:
#     st.subheader("Explanation")
#     st.markdown("Click on the image to see how each attribute contributed and hover over them to see the SHAP values")

with explanation2:
    st.subheader("Visual map")
    st.write("This might take a moment to load, please be patient")
    st.write("Click on the image to see the shap values")
    components.iframe("https://observablehq.com/embed/d177ef99668b6553@1065?cells=viewof+button%2Cchart2", scrolling=False, height=683)


with footer2:

    def is_user_active():
        if 'user_active4' in st.session_state.keys() and st.session_state['user_active4']:
            return True
        else:
            return False
    # if st.button('press here to edit'):
    if is_user_active():
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
                #st.write("question 1", q1)
                st.session_state.oocsi.send('EngD_HAII', {
                    'participant_ID': st.session_state.participantID,
                    'type of explanation': 'visualmap',
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
                    st.session_state.profileIndex =st.session_state.profileIndices[0]
                    switch_page(st.session_state.pages[st.session_state.nextPage4])
    else:
        if st.button('Continue to evaluation'):
            st.session_state['user_active4']=True
            st.experimental_rerun()



