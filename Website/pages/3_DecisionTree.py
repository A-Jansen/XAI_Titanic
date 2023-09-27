import streamlit as st
import pandas as pd
import numpy as np
from oocsi_source import OOCSI
from uuid import uuid4
from streamlit_extras.switch_page_button import switch_page
import random
import dtreeviz
import xgboost as xgb
from dtreeviz.trees import dtreeviz
from sklearn.tree import DecisionTreeClassifier
import graphviz as graphviz
from sklearn.datasets import make_moons
import base64

# import os
# os.environ["PATH"] += os.pathsep + 'D:/Program Files (x86)/Graphviz2.38/bin/'

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
    
name= st.session_state.X_test_names.loc[st.session_state.profileIndex, "Name"]



header1, header2, header3 = st.columns([1,2,1])
characteristics1, characteristics2, characteristics3 = st.columns([3,8,1])
prediction1, prediction2, prediction3 =st.columns([1,2,1])
title1, title2, title3 = st.columns([1,2,1])
explanation1, explanation2, explanation3 = st.columns([2,3,1])
footer1, footer2, footer3 =st.columns([1,2,1])


@st.cache_data(persist=True)
def loadData():
    url_traindf="https://raw.githubusercontent.com/A-Jansen/XAI_Titanic/main/Website/assets/train_df.csv"
    # train_df = pd.read_csv('/assets/train_df.csv')
    train_df=pd.read_csv(url_traindf, index_col=None)
    url_testdf="https://raw.githubusercontent.com/A-Jansen/XAI_Titanic/main/Website/assets/test_df.csv"
    test_df = pd.read_csv(url_testdf, index_col=None)
    X_train = train_df.drop("Survived", axis=1)
    Y_train = train_df["Survived"]
    X_test  = test_df.drop("PassengerId", axis=1).copy()
    return X_train, Y_train, X_test

@st.cache_resource
def trainModel(X_train,Y_train):
    model = xgb.XGBClassifier().fit(X_train, Y_train)
    return model

# @st.cache_resource
# def createTree(_model, X_train, Y_train, X_test):
#     # X, y = make_moons(n_samples=20, noise=0.25, random_state=3)
#     # treeclf = DecisionTreeClassifier(random_state=0)
#     # treeclf.fit(X, y)
#     # viz_model= dtreeviz(treeclf, X, y, target_name="Classes",
#     #     feature_names=["f0", "f1"], class_names=["c0", "c1"])
#     # clf = DecisionTreeClassifier(max_depth=3)
#     # clf.fit(X_train, Y_train)

#     # Y_pred = clf.predict(X_test)  
#     # acc_decision_tree2 = round(clf.score(X_train, Y_train) * 100, 2)
#     # viz_model = dtreeviz(clf,
#     #                      X_train, Y_train,
#     #                     feature_names=X_train.columns,
#     #                     target_name='Survived',
#     #                     class_names=['Dead', 'Alive'],
#     #                     X=X_test.iloc[1]  
#     # ) 
#     viz_model = dtreeviz(_model, 
#                          X_train, Y_train,
#                          tree_index=0,
#                          feature_names=list(X_train.columns),
#                          target_name='Survived',
#                          class_names=['Dead', 'Alive'],
#                          X=X_test.iloc[st.session_state.profileIndex],
#                         #depth_range_to_display=(0, 2),
#                         show_just_path=True,
#                         # orientation ='LR',
#                          )
#     path = "/assets/images/prediction_path" + str(st.session_state.profileIndex) +".svg"
#     viz_model.save(path) 
#     return viz_model

def render_svg(svg):
    """Renders the given svg string."""
    b64 = base64.b64encode(svg.encode('utf-8')).decode("utf-8")
    html = r'<img src="data:image/svg+xml;base64,%s"/>' % b64
    st.write(html, unsafe_allow_html=True)


with header2: #header2
    st.header("Decision Tree")
    st.markdown('''Decision tree models show the decisions made to come to a final prediction. 
    At each point it needs to go to go left or right based on the value of one of the attributes (e.g. sex or age) and below the final path it took to come to a decision is shown.
     ''')

    st.subheader(name)
    XGBmodel= trainModel(st.session_state.X_train, st.session_state.Y_train)
    # st.write("For debugging:")
    # st.write(st.session_state.participantID)
    
with characteristics2:
    # initialize list of lists
    data = st.session_state.X_test.iloc[st.session_state.profileIndex].values.reshape(1, -1)
    # Create the pandas DataFrame
    df = pd.DataFrame(data, columns=st.session_state.X_test.columns)
    st.dataframe(df)


with prediction2:
    prediction =  XGBmodel.predict(st.session_state.X_test.iloc[st.session_state.profileIndex].values.reshape(1, -1))
    probability = XGBmodel.predict_proba(st.session_state.X_test.iloc[st.session_state.profileIndex].values.reshape(1, -1))
    if prediction == 0:
        prob = round((probability[0][0]*100),2)
        st.markdown("The model predicts with {}% probability  that {}  will :red[**not survive**]".format(prob, name) )
    else:
        prob = round((probability[0][1]*100),2)
        st.markdown("The model predicts with {}% probability  that {}  will :green[**survive**]".format(prob, name) )

with title2: 
    st.subheader("Explanation - Decision Tree")


with explanation2: 
    # st.markdown('''Decision Tree model are a non-parametric supervised learning method
    #  commonly used for classification and regression.
    #  They are constructed using two kinf of elements: Nodes and branches. At each node (intersection),
    #  one of the data features is evaluated to split the observations into different paths.

    
    # At typical decision example is shown in the graph below.    
    # ''')

    # st.image('assets/Decision_tree.jpg')

    # st.markdown(''' The Root Node starts the graph. It is usually the variable that splits the more lcearly the data. 
    # Then, intermediate nodes are vsisble were different varaibales are evaluated but no final prediction is made yet. 
    # Finally, leaf nodes are present where the predicrtions (numerical of categoriacl) are made. 

    # For the Titanic dataset, the prediction will be whether the studied person survived the shipwreck.
    #  ''')

    

    # with st.spinner("Please be patient, we are generating a new explanation"):
        #viz_model = createTree(XGBmodel, st.session_state.X_train, st.session_state.Y_train, st.session_state.X_test)
    # st.image("/assets/images/prediction_path.svg", width =200, use_column_width=True)
    #viz_model.view()
     # read in svg prediction path and display
    #     path = "/assets/images/prediction_path" + str(st.session_state.profileIndex) +".svg"
    # # st.success("Done!")
    # with open(path, "r") as f:
    #     svg = f.read()
    if(st.session_state.profileIndex ==25 ):
        url= "https://raw.githubusercontent.com/A-Jansen/XAI_Titanic/main/Website/assets/images/dt_robins.svg"
    else:
        url = "https://raw.githubusercontent.com/A-Jansen/XAI_Titanic/main/Website/assets/images/dt_evans.svg"
    st.image(url)
    
    st.text("")

with explanation1: 
    st.dataframe(st.session_state.title_df.set_index('Title indices'))
    st.dataframe(st.session_state.gender_df.set_index('Gender indices'))
    st.dataframe(st.session_state.ports_df.set_index('Ports indices'))

with footer2:
    if (st.session_state.index2 < len(st.session_state.profileIndices)-1):
        if st.button("New profile"):
            st.session_state.index2 = st.session_state.index2+1
            st.session_state.profileIndex = st.session_state.profileIndices[st.session_state.index2]
            st.experimental_rerun()
    else:
        def is_user_active():
            if 'user_active2' in st.session_state.keys() and st.session_state['user_active2']:
                return True
            else:
                return False
        if is_user_active():
            # if st.button("Continue to evaluation"):
            #     st.write(" ")
            with st.form("my_form2", clear_on_submit=True):
                st.subheader("Evaluation")
                st.write("These questions only ask for your opinion about this specific explanation")

                c_load = st.radio("Please rate your mental effort required to understand this type of explanation",
                                  ["very, very low mental effort", "very low mental effort", "low mental effort",
                                   "rather low mental effort", "neither low nor high mental effort", "rather high mental effort", 
                                   "high mental effort", "very high mental effort", "very, very high mental effort"],
                                    horizontal=True)
        
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
                    st.session_state.oocsi.send('XAImethods_evaluation', {
                        'participant_ID': st.session_state.participantID,
                        'type of explanation': 'Decision tree',
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
                        st.session_state.profileIndex =st.session_state.profileIndices[0]
                        switch_page(st.session_state.pages[st.session_state.nextPage2])
        else:
            if st.button('Continue to evaluation'):
                st.session_state['user_active2']=True
                st.experimental_rerun()
