import streamlit as st
import pandas as pd
import numpy as np
import copy
from oocsi_source import OOCSI
from uuid import uuid4
from streamlit_extras.switch_page_button import switch_page
import random
# import shap
import dice_ml
from dice_ml.utils import helpers
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier


# st.markdown("""<style> 
# .stSlider {
#     padding-bottom: 20px;    
#     }
#     </style> """, 
#     unsafe_allow_html=True)

# st.session_state.Y_train
# st.session_state.X_test
# st.session_state.X_test_names


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
    
 
if 'profileIndex' not in st.session_state:
    st.session_state.profileIndex= st.session_state.profileIndices[st.session_state.index3]      

header1, header2, header3 = st.columns([1,2,1])
characteristics1, characteristics2, characteristics3 = st.columns([1,2,1])
prediction1, prediction2, prediction3 =st.columns([1,2,1])
explanation1, explanation2, explanation3 = st.columns([1,10,1])
footer1, footer2, footer3 =st.columns([1,2,1])
evaluation1, evaluation2, evaluation3 = st.columns([1,2,1])



name= st.session_state.X_test_names.loc[st.session_state.profileIndex, "Name"]


@st.cache_resource
def trainModel(X_train,Y_train):
    model_1 = RandomForestClassifier().fit(X_train, Y_train)  ## Random forest because XGBoost doesn't work with counterfactuals
    return model_1


@st.cache_resource
def getcounterfactual_values(_model,X_prediction, X_train):
    # compute counterfactual values
    url_traindf="https://raw.githubusercontent.com/A-Jansen/XAI_Titanic/main/Website/assets/train_df.csv"
    # train_df = pd.read_csv('/assets/train_df.csv')
    train_df=pd.read_csv(url_traindf, index_col=None
    continous_col=["Age", 'Fare', 'Siblings_spouses', 'Title', 'Parents_children','relatives' ]
    # test_df_counter = X_test.copy()
    # test_df_counter['Survived'] = X_prediction
    dice_data = dice_ml.Data(dataframe=train_df,continuous_features=continous_col, outcome_name='Survived')
    dice_model= dice_ml.Model(model=_model, backend="sklearn")
    explainer = dice_ml.Dice(dice_data, dice_model, method="random")
    return explainer



def Counterfactualsplot(X_test, explainer):
    e1 = explainer.generate_counterfactuals(
        X_test[1:2],total_CFs=4, desired_class="opposite",
         features_to_vary = ['Age','Pclass', 'Sex','Siblings_spouses', 'Parents_children', 'Embarked', 'relatives',  'Title']  ) ## Deck, Fare 
    e1.cf_examples_list[0].final_cfs_df.to_csv(path_or_buf=rf'assets\counterfactuals_{name}.csv', index=False)
    counter_csv = pd.read_csv(f'assets\counterfactuals_{name}.csv')
    return st.dataframe(counter_csv, width=10000)

with header2:
    st.header("Explanation - Counterfactuals")
    st.markdown('''A counterfactual explanation describes a situation where if a specific event had not occurred, the conclusion would have been different
    and a specific outcome would not have occurred. In machine learning, counterfactuals are used to explain prediction of individuals instances. The prediction
    of the model will be analysed and certain conditions/features that created this prediction will be modified to obtain an different outcome for the model.''')

    st.markdown('''As displayed in the graph below, the relation betwwen the inputs andthe prediciton is modified by the feature values that creates a simple causal
    relationshhip betwen inputs and predictions. 
''') 

    st.image('assets/counterfactual.jpg', caption = 'Causal relation between inputs and predictions', use_column_width = 'always' )

    st.markdown('''A counterfactual explanation of a prediction will then describe the smallest amount of change that is necessary to make to change the output
    prediction to a predefine one.''')
    st.subheader(name, anchor='top')
    # st.write("For debugging:")
    # st.write(st.session_state.participantID)
    random_forest= trainModel(st.session_state.X_train, st.session_state.Y_train)
    
with characteristics2:
    # initialize list of lists
    data = st.session_state.X_test.iloc[st.session_state.profileIndex].values.reshape(1, -1)
    # Create the pandas DataFrame
    df = pd.DataFrame(data, columns=st.session_state.X_test.columns)
    st.dataframe(df)


with prediction2:
    # st.header("Prediction")
    prediction =  random_forest.predict(st.session_state.X_test.iloc[st.session_state.profileIndex].values.reshape(1, -1))
    prediction_all = random_forest.predict(st.session_state.X_test.values)
    probability = random_forest.predict_proba(st.session_state.X_test.iloc[st.session_state.profileIndex].values.reshape(1, -1))
    if prediction == 0:
        prob = round((probability[0][0]*100),2)
        st.markdown("The model predicts with {}% probability  that {}  will :red[**not survive**]".format(prob, name) )
    else:
        prob = round((probability[0][1]*100),2)
        st.markdown("The model predicts with {}% probability  that {}  will :green[**survive**]".format(prob, name) )

with explanation2:
    st.subheader("Explanation")
    st.markdown("counterfactual, more text here")

    

    # with st.spinner("Please be patient, we are generating a new explanation"):
    explainer= getcounterfactual_values(random_forest, prediction_all, st.session_state.X_test)
    st.set_option('deprecation.showPyplotGlobalUse', False)
    e1=Counterfactualsplot(st.session_state.X_test, explainer)
    data_indices = pd.concat([d.reset_index(drop=True) for d in [st.session_state.ports_df, st.session_state.title_df, st.session_state.gender_df]], axis=1)
    st.dataframe(data_indices)

with footer2:
    if (st.session_state.index3 < len(st.session_state.profileIndices)-1):
        if st.button("New profile"):
            st.session_state.index3 = st.session_state.index3+1
            st.session_state.profileIndex = st.session_state.profileIndices[st.session_state.index3]
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
                        st.session_state.profileIndex =st.session_state.profileIndices[0]
                        switch_page(st.session_state.pages[st.session_state.nextPage3])
        else:
            if st.button('Continue to evaluation'):
                st.session_state['user_active3']=True
                st.experimental_rerun()

