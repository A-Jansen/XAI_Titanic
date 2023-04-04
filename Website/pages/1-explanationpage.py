import streamlit as st
from uuid import uuid4
from streamlit_extras.switch_page_button import switch_page
import random
import pandas as pd
import xgboost as xgb
import copy

header1, header2, header3 = st.columns([1,2,1])
body1, body2, body3 =st.columns([1,2,1])
footer1, footer2, footer3 =st.columns([1,2,1])

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
    st.image("https://github.com/A-Jansen/XAI_Titanic/blob/main/Website/assets/titanic.jpg")
    st.header('Explanation experiment')
    st.markdown('''In this experiment we will show you two different profiles of passengers. 
    Using Machine Learning (ML) we will show a prediction whether they would have survived the disaster. 
    This prediction is accompanied by each time a different type of explanation.''')
    st.markdown("After seeing 2 profiles, you will be asked to evaluate the explanation you have just seen.")

    st.subheader('Demographic information')
    st.markdown("Before you start with the study we would like to ask you to first answer these questions")



with footer2:
    with st.form("demographic_form", clear_on_submit=True):
        gender = st.radio("How do you identify your gender", ('Female', 'Male', 'Non-binary', 'Other', 'Prefer not to say'))
        age  = st.radio("How old are you?", ('18-25', '26-35', '36-45', '46-55', '56-65', '66-75', '75+'))
        educationlevel = st.radio("What is your highest level of education?",
        ('elementary school', 'high school', 'MBO', 'HBO', 'University'))
        st.markdown('**AI literacy**')
        st.markdown('Please rate to what extent you have the skills/knowledge listed below. 0 means that he ability is hardly or not at all pronounced, whereas a value of 10 means that the ability is very well or almost perfectly pronounced')
        q1 = st.slider('I know the most important concepts of the topic "artificial intelligence"', 0, 10)
        q2 = st.slider("I know definitions of artificial intelligence", 0, 10 )
        q3 = st.slider("I can assess what the limitations and opportunities of using an AI are", 0, 10)
        q4 = st.slider("I can assess what advantages and disadvantages the  use of an artificial intelligence entails", 0, 10)
        q5 = st.slider("I can think of new uses for AI.", 0, 10)
        q6 = st.slider("I can imagine possible future uses of AI", 0, 10)

        st.markdown('''On the next page you will see a profile of one of the passengers of the Titanic,
        a prediction of whether they would have survived and an explanation for why the model made this prediction. Have a look at this and then generate a new profile by clicking on the button.
        You can look at 2 profiles, next you will be asked to evaluate the explanation. 
        These steps will be repeated in total 4 times after which you will be asked some final questions.  ''')

        submitted = st.form_submit_button("Start the experiment")

        if submitted:
            st.session_state.oocsi.send('EngD_HAII_demographics', {
                    'participant_ID': st.session_state.participantID,
                    'gender': gender,
                    'age': age,
                    'educationLevel': educationlevel,
                    'q1': q1,
                    'q2': q2,
                    'q3': q3,
                    'q4': q4,
                    'q5': q5,
                    'q6': q6,                  
                    })
        # if st.button("Start the experiment "):

            switch_page(st.session_state.pages[st.session_state.nextPage])
