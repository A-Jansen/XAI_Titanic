import streamlit as st
import pandas as pd
import numpy as np
from oocsi_source import OOCSI
from uuid import uuid4
from streamlit_extras.switch_page_button import switch_page
import random
import shap
import xgboost
import matplotlib.pyplot as plt

# st.markdown("""<style> 
# .stSlider {
#     padding-bottom: 20px;    
#     }
#     </style> """, 
#     unsafe_allow_html=True)

#Delete this page from the array of pages to visit, this way it cannot be visited twice
if 'profile1' not in st.session_state:
    st.session_state.profiles.remove("profile1")
    st.session_state.profile1= 'deleted'
    if (len(st.session_state.profiles)>0):
        st.session_state.nextPage1 = random.randint(0, len(st.session_state.profiles)-1)
        st.session_state.lastQuestion= 'no'
    else:
        st.session_state.lastQuestion= 'yes'
    
    

header = st.container()
characteristics = st.container()
prediction = st.container()
explanation = st.container()
evaluation = st.container()




@st.cache_resource
def trainModel(X_train, Y_train, X_test):
    # train XGBoost model

    model = xgboost.XGBClassifier().fit(X_train, Y_train)

    # compute SHAP values
    explainer = shap.Explainer(model, X_test)
    shap_values = explainer(X_test)
    return shap_values

@st.cache_resource
def shapPlot(X_test, _shap_values):
    # model.predict(X_test.iloc[0].values.reshape(1, -1))
    return shap.plots.waterfall(shap_values[0])


with header:
    st.header("Mr. James Kelly")
    st.write("For debugging:")
    st.write(st.session_state.participantID)
    
with characteristics:
    # initialize list of lists
    data = [['Male', '34.5', '3', 'Mr', '0', '40', '1', 'Southampton']]
    
    # Create the pandas DataFrame
    df = pd.DataFrame(data, columns=['Sex','Age', 'Class', 'Title', 'Number of relatives', 'Fare', 'Deck' , 'City of embarkment'])
  

    st.table(df)


with prediction:
    st.header("Prediction")
    st.markdown("The ML model predicts that Mr. James Kelly will **not** survive")

with explanation:
    shap_values= trainModel(st.session_state.X_train, st.session_state.Y_train, st.session_state.X_test)
    # plot = shapPlot(X_test, shap_values)

    st.subheader("Explanation")
    st.markdown("SHAP waterfall plot, more text here")
    st.set_option('deprecation.showPyplotGlobalUse', False)
    fig = shap.plots.waterfall(shap_values[0])
    st.pyplot(fig, bbox_inches='tight')



with evaluation:
    # finished1 = st.checkbox('I am finished looking at the explanation, continue to the questions')
    # if finished1:
    with st.form("my_form1", clear_on_submit=True):
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
                'profile': 'Mr James Kelly',
                'type of explanation': 'SHAP',
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
                    switch_page(st.session_state.profiles[st.session_state.nextPage1])