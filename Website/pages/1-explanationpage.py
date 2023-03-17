import streamlit as st
from uuid import uuid4
from streamlit_extras.switch_page_button import switch_page
import random
import pandas as pd
import xgboost as xgb


header = st.container()
body = st.container()
footer =st.container()

if 'nextPage' not in st.session_state:
    st.session_state.nextPage = random.randint(0, len(st.session_state.profiles)-1)
st.write(st.session_state.nextPage)


@st.cache_data
def loadData():
    train_df = pd.read_csv('assets/train_df.csv')
    test_df = pd.read_csv('assets/test_df.csv')
    test_with_names = pd.read_csv('assets/test_with_names.csv')
    X_train = train_df.drop("Survived", axis=1)
    Y_train = train_df["Survived"]
    X_test  = test_df.drop("PassengerId", axis=1).copy()
    X_test_names = test_with_names.copy()
    return X_train, Y_train, X_test, X_test_names

if 'X_train' not in st.session_state:
    st.session_state.X_train, st.session_state.Y_train, st.session_state.X_test, st.session_state.X_test_names= loadData()
    # st.dataframe(st.session_state.X_train)



with header:
    st.title("Who survived and why?")
    st.write("For debugging:")
    st.write(st.session_state.participantID)
    # X_train, Y_train, X_test= loadData()

with body:
    st.header("The Titanic")
    st.markdown("In the year 1912, the Titanic left from Southampton to New York City, but it never arrived. On April 15, it crashed into an iceberg and sunk. Of the estimated 2,224 passengers and crew aboard, more than 1,500 died, making it the deadliest sinking of a single ship up to that time. ")
    st.image('assets/titanic.jpg')
    st.header('Explanation experiment')
    st.markdown("In this experiment we will show you different profiles of passengers who were on the titanic(?). Using Machine Learning (ML) we will show a prediction whether they would have survived the disaster. This prediction is accompanied by each time a different type of explanation.")
    st.markdown(" After each profile, you will get a few questions about the explanation that was given. In total there are # questions.")


with footer:
    if st.button("Start the experiment "):
        switch_page(st.session_state.profiles[st.session_state.nextPage])
