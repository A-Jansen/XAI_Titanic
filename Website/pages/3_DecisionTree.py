import streamlit as st
import pandas as pd
import numpy as np
from oocsi_source import OOCSI
from uuid import uuid4
from streamlit_extras.switch_page_button import switch_page
import random
from dtreeviz.trees import *
from sklearn.tree import DecisionTreeClassifier
import graphviz as graphviz
from sklearn.datasets import make_moons
import base64

# st.markdown("""<style> 
# .stSlider {
#     padding-bottom: 20px;    
#     }
#     </style> """, 
#     unsafe_allow_html=True)

#Delete this page from the array of pages to visit, this way it cannot be visited twice
if 'profile2' not in st.session_state:
    st.session_state.pages.remove("DecisionTree")
    st.session_state.profile2= 'deleted'
    if (len(st.session_state.pages)>0):
        st.session_state.nextPage2 = random.randint(0, len(st.session_state.pages)-1)
        st.session_state.lastQuestion= 'no'
    else:
        st.session_state.lastQuestion= 'yes'


if 'index2' not in st.session_state:
    st.session_state.index2= 0    


if 'profileIndex' not in st.session_state:
    st.session_state.profileIndex= st.session_state.profileIndices[st.session_state.index2]       
    
nameArray =st.session_state.X_test_names.loc[st.session_state.profileIndex, "Name"].split(',')
name= nameArray[1]+" "+ nameArray[0]


header = st.container()
characteristics = st.container()
prediction = st.container()
explanation = st.container()
footer = st.container()
evaluation = st.container()

@st.cache_resource
def loadData():
    train_df = pd.read_csv('assets/train_df.csv')
    test_df = pd.read_csv('assets/test_df.csv')
    X_train = train_df.drop("Survived", axis=1)
    Y_train = train_df["Survived"]
    X_test  = test_df.drop("PassengerId", axis=1).copy()
    return X_train, Y_train, X_test

@st.cache_resource
def trainModel(X_train,Y_train):
    model = xgb.XGBClassifier().fit(X_train, Y_train)
    return model

@st.cache_resource
def createTree(X_train, Y_train, X_test):
    # X, y = make_moons(n_samples=20, noise=0.25, random_state=3)
    # treeclf = DecisionTreeClassifier(random_state=0)
    # treeclf.fit(X, y)
    # viz_model= dtreeviz(treeclf, X, y, target_name="Classes",
    #     feature_names=["f0", "f1"], class_names=["c0", "c1"])
    clf = DecisionTreeClassifier(max_depth=3)
    clf.fit(X_train, Y_train)
    # Y_pred = clf.predict(X_test)  
    # acc_decision_tree2 = round(clf.score(X_train, Y_train) * 100, 2)
    viz_model = dtreeviz(clf,
                         X_train, Y_train,
                        feature_names=X_train.columns,
                        target_name='Survived',
                        class_names=['Dead', 'Alive'],
                        X=X_test.iloc[1]  
    ) 
    viz_model.save("/assets/images/prediction_path.svg") 
    return viz_model

def render_svg(svg):
    """Renders the given svg string."""
    b64 = base64.b64encode(svg.encode('utf-8')).decode("utf-8")
    html = r'<img src="data:image/svg+xml;base64,%s"/>' % b64
    st.write(html, unsafe_allow_html=True)


with header:
    st.header(name)
    XGBmodel= trainModel(st.session_state.X_train, st.session_state.Y_train)
    # st.write("For debugging:")
    # st.write(st.session_state.participantID)
    
with characteristics:
    # initialize list of lists
    data = st.session_state.X_test.iloc[st.session_state.profileIndex].values.reshape(1, -1)
    # Create the pandas DataFrame
    df = pd.DataFrame(data, columns=st.session_state.X_test.columns)
    st.dataframe(df)


with prediction:
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
    st.markdown("One of the four types of predictions")
    # X_train, Y_train, X_test= loadData()
    viz_model = createTree(st.session_state.X_train, st.session_state.Y_train, st.session_state.X_test)
    # st.image("/assets/images/prediction_path.svg", width =200, use_column_width=True)
    #viz_model.view()
     # read in svg prediction path and display
    with open("/assets/images/prediction_path.svg", "r") as f:
        svg = f.read()
    render_svg(svg)
    st.text("")

with footer:
    if (st.session_state.index2 < len(st.session_state.profileIndices)-1):
        if st.button("New profile"):
            st.session_state.index2 = st.session_state.index2+1
            st.session_state.profileIndex = st.session_state.profileIndices[st.session_state.index2]
            st.experimental_rerun()
    else:
        st.markdown("You have reached the end of the profiles :disappointed_relieved:")
        if st.button("Continue to evaluation"):
            st.write(" ")
        with st.form("my_form2", clear_on_submit=True):
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
                    'type of explanation': 'Decision tree',
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
                    switch_page(st.session_state.pages[st.session_state.nextPage2])
