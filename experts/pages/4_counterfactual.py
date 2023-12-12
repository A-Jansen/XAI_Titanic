import streamlit as st
import pandas as pd
import numpy as np
import copy
from oocsi_source import OOCSI
from uuid import uuid4
from streamlit_extras.switch_page_button import switch_page
import random
import datetime
import dice_ml
from dice_ml.utils import helpers
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime


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
            "duration_seconds": page_duration.total_seconds(), 
            'participant_ID': st.session_state.participantID
        }
        st.session_state.oocsi.send('Time_XAI', data)

st.session_state.current_page_title = "Counterfactual"
page_start_time = None
record_page_start_time()

#Delete this page from the array of pages to visit, this way it cannot be visited twice
if 'profile3' not in st.session_state:
    st.session_state.pages.remove("counterfactual")
    st.session_state.profile3= 'deleted'
    if (len(st.session_state.pages)>0):
        st.session_state.nextPage3 = random.randint(0, len(st.session_state.pages)-1)
        st.session_state.lastQuestion= 'no'
    else:
        st.session_state.lastQuestion= 'yes'


if 'index3' not in st.session_state:
    st.session_state.index3= 0    
    
 
# if 'profileIndex' not in st.session_state:
st.session_state.profileIndex= st.session_state.profileIndices_counter[st.session_state.index3]      

header1, header2, header3 = st.columns([1,2,1])
characteristics1, characteristics2, characteristics3 = st.columns([3,8,1])
prediction1, prediction2, prediction3 =st.columns([1,2,1])
title1, title2, title3 = st.columns([1,2,1])
explanation1, explanation2, explanation3 = st.columns([1,2,1])
footer1, footer2, footer3 =st.columns([1,2,1])
evaluation1, evaluation2, evaluation3 = st.columns([1,2,1])
presentation1, presentation2, presentation3 = st.columns([2,2,2])

# sex_mapping = {0: 'Male', 1: 'Female'}
# title_mapping = {1: 'Mr', 2: 'Miss', 3: 'Mrs', 4: 'Master', 5: 'Rare'}
# port_mapping = {0: 'Southampton', 1: 'Cherbourg', 2: 'Queenstown'}

name= st.session_state.X_test_names.loc[st.session_state.profileIndex, "Name"]


@st.cache_resource
def trainModelRF(X_train,Y_train):
    model_1 = RandomForestClassifier().fit(X_train, Y_train)  ## Random forest because XGBoost doesn't work with counterfactuals
    return model_1

@st.cache_resource
def trainModel(X_train,Y_train):
    model = xgb.XGBClassifier().fit(X_train, Y_train)
    return model

@st.cache_resource
def getcounterfactual_values(_model,X_prediction, X_train):
    # compute counterfactual values
    url_traindf="https://raw.githubusercontent.com/A-Jansen/XAI_Titanic/main/experts/assets/train_df.csv"
    train_df=pd.read_csv(url_traindf, index_col=None)
    continuous_col=["Age", 'Fare']
    dice_data = dice_ml.Data(dataframe=train_df,continuous_features=continuous_col, outcome_name='Survived')
    dice_model= dice_ml.Model(model=_model, backend="sklearn")
    explainer = dice_ml.Dice(dice_data, dice_model, method="random")
    return explainer

def highlight_changes(val):
    color = 'background-color: yellow' if isinstance(val, float) else ''
    return color

def Counterfactualsplot(X_test, explainer):
    life_mapping = {0: 'Not Survived', 1: 'Survived'}  
    sex_mapping = {0: 'Male', 1: 'Female', '0': 'Male'}
    title_mapping = {1: 'Mr','1': 'Mr', 2: 'Miss', 3: 'Mrs', 4: 'Master', 5: 'Rare'}
    port_mapping = {0: 'Southampton', 1: 'Cherbourg', 2: 'Queenstown'}
    e1 = explainer.generate_counterfactuals(X_test[st.session_state.profileIndex:st.session_state.profileIndex+1], total_CFs=3, 
                                            features_to_vary = ['Age','Siblings_spouses','Title','Parents_children','relatives','Pclass','Embarked','Deck','Sex'], desired_class="opposite")
    
    counterfactual_instance = e1.cf_examples_list[0].final_cfs_df
    counterfactual_instance['Sex'] = counterfactual_instance['Sex'].replace(sex_mapping)
    counterfactual_instance['Title'] = counterfactual_instance['Title'].replace(title_mapping)
    counterfactual_instance['Embarked'] = counterfactual_instance['Embarked'].replace(port_mapping) 
# <<<<<<< Updated upstream
#     counterfactual_instance['Survived'] = counterfactual_instance['Survived'].replace(life_mapping) 
#     return st.dataframe(counterfactual_instance.set_index(counterfactual_instance.columns[0]), use_container_width= True)
# =======
    return st.dataframe(counterfactual_instance.set_index(df.columns[0]), use_container_width= False)
    # st.dataframe(df.set_index(df.columns[0]), use_container_width= False)
# >>>>>>> Stashed changes


with header2:
    st.header("Counterfactuals")
    st.markdown('''A counterfactual is used to showcase which attributes (e.g. sex or age) would need to change to get the opposite outcome, i.e. to survive when the prediction is 'not survive'. 
    Multiple changes are shown but all have the opposite outcome from the current prediction. ''')

    st.subheader(name, anchor='top')
    random_forest= trainModelRF(st.session_state.X_train, st.session_state.Y_train)
    XGBmodel= trainModel(st.session_state.X_train, st.session_state.Y_train)
    
with characteristics2:
    sex_mapping = {0: 'Male', 1: 'Female'}
    title_mapping = {1: 'Mr', 2: 'Miss', 3: 'Mrs', 4: 'Master', 5: 'Rare'}
    port_mapping = {0: 'Southampton', 1: 'Cherbourg', 2: 'Queenstown'}
    # initialize list of lists
    data = st.session_state.X_test.iloc[st.session_state.profileIndex].values.reshape(1, -1)
    # Create the pandas DataFrame
    df = pd.DataFrame(data, columns=st.session_state.X_test.columns)
    df['Sex'] = df['Sex'].replace(sex_mapping)
    df['Title'] = df['Title'].replace(title_mapping)
    df['Embarked'] = df['Embarked'].replace(port_mapping)  
    st.dataframe(df.set_index(df.columns[0]), use_container_width= False)



with prediction2:
    # st.header("Prediction")
    prediction =  random_forest.predict(st.session_state.X_test.iloc[st.session_state.profileIndex].values.reshape(1, -1))
    prediction_all = random_forest.predict(st.session_state.X_test.values)
    probability = XGBmodel.predict_proba(st.session_state.X_test.iloc[st.session_state.profileIndex].values.reshape(1, -1))
    if prediction == 0:
        prob = round((probability[0][0]*100),2)
        st.markdown("The model predicts with {}% probability  that {}  will :red[**not survive**]".format(prob, name) )
    else:
        prob = round((probability[0][1]*100),2)
        st.markdown("The model predicts with {}% probability  that {}  will :green[**survive**]".format(prob, name) )

with title2: 
    st.subheader("Explanation - counterfactuals ")


with explanation2:
    if prediction == 0:
        st.markdown("The model predicts  that {}  will :red[**not survive**], the counterfactual will show the change necessary to obtain the :green[**opposite outcome**]".format(name) )
    else:
        st.markdown("The model predicts  that {}  will :green[**survive**], the counterfactual will show the change necessary to obtain the :red[**opposite outcome**]".format(name) )
    # explainer= getcounterfactual_values(random_forest, prediction_all, st.session_state.X_test)
    st.set_option('deprecation.showPyplotGlobalUse', False)
    # e1=Counterfactualsplot(st.session_state.X_test, explainer)
    # data_indices = pd.concat([d.reset_index(drop=True) for d in [st.session_state.ports_df, st.session_state.title_df, st.session_state.gender_df]], axis=1)
    # st.dataframe(data_indices)

    if(st.session_state.profileIndex ==26 ):
        url= "https://raw.githubusercontent.com/A-Jansen/XAI_Titanic/main/experts/assets/images/counter_26_helene.png"
    elif(st.session_state.profileIndex ==69 ):
        url = "https://raw.githubusercontent.com/A-Jansen/XAI_Titanic/main/experts/assets/images/counter_69_mark.png"
    elif(st.session_state.profileIndex ==113 ):
        url = "https://raw.githubusercontent.com/A-Jansen/XAI_Titanic/main/experts/assets/images/counter_113_kath.png"
    else: #57
        url = "https://raw.githubusercontent.com/A-Jansen/XAI_Titanic/main/experts/assets/images/counter_57_olaus.png"

    st.image(url, width=700)
    st.text("")

    
with footer2:
    if (st.session_state.index3 < len(st.session_state.profileIndices_counter)-1):
        if st.button("New profile"):
            st.session_state.index3 = st.session_state.index3+1
            st.session_state.profileIndex = st.session_state.profileIndices_counter[st.session_state.index3]
            st.experimental_rerun()
    else:
        def is_user_active():
            if 'user_active3' in st.session_state.keys() and st.session_state['user_active3']:
                return True
            else:
                return False
        if is_user_active():
        # if st.button("Continue to evaluation"):
        #     st.write(" ")
            with st.form("my_form3", clear_on_submit=True):
                st.subheader("Evaluation")
                st.write("These questions only ask for your opinion about this specific explanation")
                c_load = st.select_slider('**1**- Please rate your mental effort required to understand this type of explanation',
                                          options=["very, very low mental effort", "very low mental effort", "low mental effort",
                                   "rather low mental effort", "neither low nor high mental effort", "rather high mental effort", 
                                   "high mental effort", "very high mental effort", "very, very high mental effort"])

                q1 = st.select_slider(
                '**2**- From the explanation, I **understand** how the model works:',
                options=['totally disagree', 'disagree', 'neutral' , 'agree', 'totally agree'])

                q2 = st.select_slider(
                '**3**- This explanation of how the model works is **satisfying**:',
                options=['totally disagree', 'disagree', 'neutral' , 'agree', 'totally agree'])

                q3 = st.select_slider(
                '**4**- This explanation of how the model works has **sufficient detail**:',
                options=['totally disagree', 'disagree', 'neutral' , 'agree', 'totally agree'])

                q4 = st.select_slider(
                '**5**- This explanation of how the model works seems **complete**:',
                options=['totally disagree', 'disagree', 'neutral' , 'agree', 'totally agree'])

                q5 = st.select_slider(
                '**6**- This explanation of how the model works **tells me how to use the model**:',
                options=['totally disagree', 'disagree', 'neutral' , 'agree', 'totally agree'])

                q6 = st.select_slider(
                '**7**- This explanation of how the model works is **useful to my goals**:',
                options=['totally disagree', 'disagree', 'neutral' , 'agree', 'totally agree'])

                q7 = st.select_slider(
                '**8**- This explanation of the model shows me how **accurate** the model is:',
                options=['totally disagree', 'disagree', 'neutral' , 'agree', 'totally agree'])

                q8 = st.select_slider(
                '**9**- This explanation lets me judge when I should **trust and not trust** the model:',
                options=['totally disagree', 'disagree', 'neutral' , 'agree', 'totally agree'])
                # Every form must have a submit button.
                submitted = st.form_submit_button("Submit")
                if submitted:
                    if page_start_time:
                        record_page_duration_and_send()    
                    # record_page_start_time()
                    # st.write("question 1", q1)
                    st.session_state.oocsi.send('XAImethods_evaluation', {
                        'participant_ID': st.session_state.participantID,
                        'type of explanation': 'counterfactual',
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
                        switch_page(st.session_state.pages[st.session_state.nextPage3])
        else:
            if st.button('Continue to evaluation'):
                st.session_state['user_active3']=True
                st.experimental_rerun()

