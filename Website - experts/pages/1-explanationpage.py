import streamlit as st
from uuid import uuid4
from streamlit_extras.switch_page_button import switch_page
import random
import pandas as pd
import datetime
import xgboost as xgb
import copy
from PIL import Image
from datetime import datetime, timedelta
import numpy as np

header1, header2, header3 = st.columns([1,2,1])
body1, body2, body3 =st.columns([1,2,1])
footer1, footer2, footer3 =st.columns([1,2,1])


def record_page_start_time():
    global page_start_time
    page_start_time = datetime.now()

# Function to record page duration and send to Data Foundry
def record_page_duration_and_send_explanation():
    current_page_title = st.session_state.current_page_title
    if page_start_time:
        page_end_time = datetime.now()
        page_duration = page_end_time - page_start_time
        # duration_str = str(page_duration)
        # hours, minutes, seconds = map(float, duration_str.split(':'))
        # total_seconds = hours * 3600 + minutes * 60 + seconds*1000
        # # st.write(f"{total_seconds}")
        # if total_seconds < 80: 
        #     return st.write(f"Have you really read carfelly all the instruction, especially the feature table?")

        st.write(f"Time spent on {current_page_title}: {page_duration}")
            # Send data to Data Foundry via OOCSI
        data = {
            "page_name": current_page_title,
            "duration_seconds": page_duration.total_seconds()
        }
        st.session_state.oocsi.send('Time_XAI', data)

st.session_state.current_page_title = "Explanantion Page"
page_start_time = None
record_page_start_time()

if 'nextPage' not in st.session_state:
    st.session_state.nextPage = random.randint(0, len(st.session_state.pages)-1)
# st.write(st.session_state.nextPage)


@st.cache_data(persist=True)
def loadData():
    url_traindf="https://raw.githubusercontent.com/A-Jansen/XAI_Titanic/main/Website/assets/train_df.csv"
    # train_df = pd.read_csv('/assets/train_df.csv')
    train_df=pd.read_csv(url_traindf, index_col=None)
    url_testdf="https://raw.githubusercontent.com/A-Jansen/XAI_Titanic/main/Website/assets/test_df.csv"
    test_df = pd.read_csv(url_testdf, index_col=None)
    url_testnames = "https://raw.githubusercontent.com/A-Jansen/XAI_Titanic/main/Website/assets/test_with_names.csv"
    test_with_names = pd.read_csv(url_testnames, index_col=None)
   # test_with_names.drop('PassengerId', axis=1, inplace=True)
    X_train = train_df.drop("Survived", axis=1)
    Y_train = train_df["Survived"]
    X_test = test_df.drop('PassengerId', axis=1)
    X_test_names = test_with_names.copy()
    title_df = pd.DataFrame({'Title indices': [1,2,3,4,5],
                             'Title': ['Mr', 'Miss', 'Mrs', 'Master', 'Rare'] })
    gender_df = pd.DataFrame({'Gender indices': [0,1],
                             'Sex': ['Male', 'Female'] })
    ports_df = pd.DataFrame({'Ports indices': [0,1,2],
                             'Embarked': ['Southampton', 'Cherbourg', 'Quenstown'] })
    return X_train, Y_train, X_test, X_test_names, title_df, gender_df, ports_df, train_df

if 'X_train' not in st.session_state:
    st.session_state.X_train, st.session_state.Y_train, st.session_state.X_test, st.session_state.X_test_names, st.session_state.title_df, st.session_state.gender_df, st.session_state.ports_df, st.session_state.train_df = loadData()
    # st.dataframe(st.session_state.X_train)
    # st.session_state.X_train, st.session_state.Y_train, st.session_state.X_test, st.session_state.X_test_names= loadData()




with header2:
    st.title("Who survived and why?")
    # st.dataframe(st.session_state.X_train)
    # st.write("For debugging:")
    # st.write(st.session_state.participantID)
    # X_train, Y_train, X_test= loadData()

with body2:
    st.header("The Titanic")
    st.markdown("In the year 1912, the Titanic left from Southampton to New York City, but it never arrived. On April 15, it crashed into an iceberg and sunk. Of the estimated 2,224 passengers and crew aboard, more than 1,500 died, making it the deadliest sinking of a single ship up to that time. ")
    st.image('https://github.com/A-Jansen/XAI_Titanic/blob/main/Website/assets/titanic.jpg?raw=true')

    st.header('Explanation experiment')
    st.markdown('''In this experiment we will show you four different profiles of passengers. 
    Using Machine Learning (ML) we will show a prediction whether they would have survived the disaster. 
    This prediction is accompanied by each time a different type of explanation.''')
    st.markdown("After seeing four profiles, you will be asked to evaluate the explanation you have just seen.")
    
    st.subheader('Model')
    st.markdown(''' The same ML model is used to generate the predictions of who survived and who did not. 
                This model is used to generate all of the four types of explanations that you will see during the experiment. 
                ''')
    
    st.subheader('Features')
    st.markdown('''We know certain \"features\" of the passengers that embarked the Titanic. A feature describes something about them, for example their age or how much they paid for the ticket. 
                These features are used by the ML model to predict whether someone would survive or not. The following features are used.
                In the following explanations you will see these coming back.''')
    df = pd.DataFrame({'Feature':['pclass','Sex','Age', 'Title', 'Siblings_spouses', 'Parents_children', 'Relatives', 'Fare', 'Embarked', 'Deck'],
                       'Description':['Gives the ticket class (1st, 2nd or 3rd). Is a proxy for socioeconomic status',
                                      'Male or Female passenger',
                                      'Age of the passenger',
                                      'Title of the passenger (Mr, Miss, Mrs, Master, rare )',
                                      'Number of siblings and spouses aboard the Titanic',
                                      'Number of parents and children aboard the Titanic',
                                      'Total number of relatives',
                                      'Price of the ticket (no currency indicated)',
                                      'Part of embarkation between Cherbourg, Queenstown and Southampton',
                                      'The deck on which the passenger\'s cabin was located']})
    st.dataframe(df.set_index(df.columns[0]), use_container_width= True)

    # st.subheader('Demographic information')
    # st.markdown("Before you start with the study we would like to ask you to first answer these questions")

    feature_explanation = st.text_input("Please shortly explain what a feature is and give an example in the case of the Titanic dataset")

with footer2:
    if st.button("Start the experiment "):
        if page_start_time:
            record_page_duration_and_send_explanation()    
        record_page_start_time()
        st.session_state.oocsi.send('XAImethods_attentioncheck', {
            'participant_ID': st.session_state.participantID,
            'feature_explanation': feature_explanation,
            })
        switch_page(st.session_state.pages[st.session_state.nextPage])
