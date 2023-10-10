import streamlit as st
import pandas as pd
import numpy as np
from oocsi_source import OOCSI
from uuid import uuid4
from streamlit_extras.switch_page_button import switch_page
import random
import datetime
import shap
from IPython.display import display_html
import xgboost as xgb
import matplotlib.pyplot as plt
from datetime import datetime

# st.markdown("""<style> 
# .stSlider {
#     padding-bottom: 20px;    
#     }
#     </style> """, 
#     unsafe_allow_html=True)

def record_page_start_time():
    global page_start_time
    page_start_time = datetime.now()

# Function to record page duration and send to Data Foundry
def record_page_duration_and_send():
    current_page_title = st.session_state.current_page_title
    if page_start_time:
        page_end_time = datetime.now()
        page_duration = page_end_time - page_start_time
        st.write(f"Time spent on {current_page_title}: {page_duration}")
        
        # Send data to Data Foundry via OOCSI
        data = {
            "page_name": current_page_title,
            "duration_seconds": page_duration.total_seconds()
        }
        st.session_state.oocsi.send('Time_XAI', data)

st.session_state.current_page_title = "Shap"
page_start_time = None
record_page_start_time()

#Delete this page from the array of pages to visit, this way it cannot be visited twice
if 'profile1' not in st.session_state:
    st.session_state.pages.remove("SHAP")
    st.session_state.profile1= 'deleted'
    if (len(st.session_state.pages)>0):
        st.session_state.nextPage1 = random.randint(0, len(st.session_state.pages)-1)
        st.session_state.lastQuestion= 'no'
    else:
        st.session_state.lastQuestion= 'yes'


if 'index1' not in st.session_state:
    st.session_state.index1= 0    



# if 'profileIndex' not in st.session_state:
st.session_state.profileIndex= st.session_state.profileIndices_SHAP[st.session_state.index1]   


header1, header2, header3 = st.columns([1,2,1])
characteristics1, characteristics2, characteristics3 = st.columns([3,8,1])
prediction1, prediction2, prediction3 =st.columns([1,2,1])
title1, title2, title3 = st.columns([1,2,1])
explanation1, explanation2, explanation3 = st.columns([1.5,4.5,1])
footer1, footer2, footer3 =st.columns([1,2,1])




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
    st.header('Explanation - SHAP Values')
    st.markdown(''' One form of explanations is using SHAP values, these values indicate how much each attribute (e.g. sex or age) contributed to the prediction. 
    Blue bars represent a contribution to a negative prediction (dead) and red bars a contribution towards the positive outcome (survived).

    ''')
    st.subheader(name, anchor='top')
    # st.write("For debugging:")
    # st.write(st.session_state.participantID)
    XGBmodel= trainModel(st.session_state.X_train, st.session_state.Y_train)
    
with characteristics2:
    # initialize list of lists
    data = st.session_state.X_test.iloc[st.session_state.profileIndex].values.reshape(1, -1)
    # Create the pandas DataFrame
    df = pd.DataFrame(data, columns=st.session_state.X_test.columns)
    st.dataframe(df.style.hide_index())
    # df_values = df.values.tolist()
    # st.table(df_values)    

with prediction2:
    # st.header("Prediction")
    prediction =  XGBmodel.predict(st.session_state.X_test.iloc[st.session_state.profileIndex].values.reshape(1, -1))
    probability = XGBmodel.predict_proba(st.session_state.X_test.iloc[st.session_state.profileIndex].values.reshape(1, -1))

    if prediction == 0:
        prob = round((probability[0][0]*100),2)
        st.markdown("The model predicts with {}% probability  that {}  will :red[**not survive**]".format(prob, name) )

    else:
        prob = round((probability[0][1]*100),2)
        st.markdown("The model predicts with {}% probability  that {}  will :green[**survive**]".format(prob, name) )

with explanation1: 
    st.dataframe(st.session_state.title_df.set_index('Title indices'))
    st.dataframe(st.session_state.gender_df.set_index('Gender indices'))
    st.dataframe(st.session_state.ports_df.set_index('Ports indices'))

with title2: 
    st.subheader("Explanation")

with explanation2:
    # with st.spinner("Please be patient, we are generating a new explanation"):
    shap_values= getSHAPvalues(XGBmodel, st.session_state.X_train, st.session_state.Y_train, st.session_state.X_test)
    st.set_option('deprecation.showPyplotGlobalUse', False)
    fig = shap.plots.waterfall(shap_values[st.session_state.profileIndex])
    st.pyplot(fig, bbox_inches='tight')
    data_indices = pd.concat([d.reset_index(drop=True) for d in [st.session_state.ports_df, st.session_state.title_df, st.session_state.gender_df]], axis=1)
    # st.dataframe(st.session_state.ports_df)
    # st.dataframe(st.session_state.title_df)
    # st.dataframe(st.session_state.gender_df)
    # st.dataframe(data_indices)



with footer2:

    if (st.session_state.index1 < len(st.session_state.profileIndices_SHAP)-1):
        if st.button("New profile"):
  
            st.session_state.index1 = st.session_state.index1+1
            st.session_state.profileIndex = st.session_state.profileIndices_SHAP[st.session_state.index1]
            st.experimental_rerun()
    else:
        def is_user_active():
            if 'user_active1' in st.session_state.keys() and st.session_state['user_active1']:
                return True
            else:
                return False
        if is_user_active():
        # st.markdown("You have reached the end of the profiles")
        # if st.button("Continue to evaluation"):
        #     st.write(" ")
            with st.form("my_form1", clear_on_submit=True):
                st.subheader("Evaluation")
                st.write("These questions only ask for your opinion about this specific explanation")
                c_load = st.radio("Please rate your mental effort required to understand this type of explanation",
                                  ["very, very low mental effort", "very low mental effort", "low mental effort",
                                   "rather low mental effort", "neither low nor high mental effort", "rather high mental effort", 
                                   "high mental effort", "very high mental effort", "very, very high mental effort"],
                                    horizontal=False)

                q1 = st.select_slider(
                '**1**- From the explanation, I **understand** how the model works:',
                options=['totally disagree', 'disagree', 'neutral' , 'agree', 'totally agree'])

                q2 = st.select_slider(
                '**2**- This explanation of how the model works is **satisfying**:',
                options=['totally disagree', 'disagree', 'neutral' , 'agree', 'totally agree'])

                q3 = st.select_slider(
                '**3**- This explanation of how the model works has **sufficient detail**:',
                options=['totally disagree', 'disagree', 'neutral' , 'agree', 'totally agree'])

                q4 = st.select_slider(
                '**4**- This explanation of how the model works seems **complete**:',
                options=['totally disagree', 'disagree', 'neutral' , 'agree', 'totally agree'])

                q5 = st.select_slider(
                '**5**- This explanation of how the model works **tells me how to use it**:',
                options=['totally disagree', 'disagree', 'neutral' , 'agree', 'totally agree'])

                q6 = st.select_slider(
                '**6**- This explanation of how the model works is **useful to my goals**:',
                options=['totally disagree', 'disagree', 'neutral' , 'agree', 'totally agree'])

                q7 = st.select_slider(
                '**7**- This explanation of the model shows me how **accurate** the model is:',
                options=['totally disagree', 'disagree', 'neutral' , 'agree', 'totally agree'])

                q8 = st.select_slider(
                '**8**- This explanation lets me judge when I should **trust and not trust** the model:',
                options=['totally disagree', 'disagree', 'neutral' , 'agree', 'totally agree'])

                # Every form must have a submit button.
                submitted = st.form_submit_button("Submit")
                if submitted:
                    if page_start_time:
                        record_page_duration_and_send()    
                    st.write("question 1", q1)
                    st.session_state.oocsi.send('XAImethods_evaluation', {
                        'participant_ID': st.session_state.participantID,
                        'type of explanation': 'SHAP',
                        'cognitive load': c_load,
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
                        # st.session_state.profileIndex =st.session_state.profileIndices[0]
                        switch_page(st.session_state.pages[st.session_state.nextPage1])
        else:
            if st.button('Continue to evaluation'):
                st.session_state['user_active1']=True
                st.experimental_rerun()
