import streamlit as st
import streamlit.components.v1 as components
from streamlit_extras.switch_page_button import switch_page
from oocsi_source import OOCSI

header1, header2, header3 = st.columns([1,4,1])
image1, image2, image3 = st.columns([1,50,1])
body1, body2, body3 =st.columns([1,50,1])


with header2:
    st.title("Comparing the different methods")
    st.markdown("This is the final section of this experiment, please rate and compare the different methods")

with image2:
    st.image('https://github.com/A-Jansen/XAI_Titanic/blob/main/Website/assets/images/overview%20methods.png?raw=true')

with body2:
    with st.form("my_form"):
        st.markdown("**Demograhpic information**")
        gender = st.radio("How do you identify your gender", ('Female', 'Male', 'Non-binary', 'Other', 'Prefer not to say'))
        age  = st.radio("How old are you?", ('18-25', '26-35', '36-45', '46-55', '56-65', '66-75', '75+'))
        educationlevel = st.radio("What is your highest level of education?",
        ('elementary school', 'high school', 'MBO', 'HBO', 'University'))
        st.markdown('**AI literacy**')
        st.markdown("Please select the right answer at the multiple choice questions below. \
                    A correct answer is awarded with +1 point, an incorrect answer -1 point and the \"I don't know option\" 0 points.")
    
        # socio1 = st.radio(
        # "AI was first mention in",
        # ["The 2000s", 
        #  "The 1950s", 
        #  "The 1880s", 
        #  "The 1980s",
        #  "I don't know"], index =4)

        # socio2 = st.radio(
        # "How are human and artificial intelligence related?",
        # ["They are the same, concerning strengths and weaknesses", 
        #  "They predict each other", 
        #  "Their strengths and weaknesses converge", 
        #  "They are different, each has its own strengths and weaknesses",
        #  "I don't know"], index =4)

        # socio3 = st.radio(
        # "AI research",
        # ["happens in an interdisciplinary field including multiple technologies ", 
        #  "refers to one specific AI technology", 
        #  "is only fiction at this point in time ", 
        #  "revolves predominantly around optimization ",
        #  "I don't know"], index =4)

        # socio4 = st.radio(
        # "What is a possible risk for humans of AI technology",
        # ["Digital assistants take over self-driving cars", 
        #  "Voice generators make people unlearn natural languages", 
        #  "Image generator break the rule of art ", 
        #  "Deep fakes render videos unattributable",
        #  "I don't know"], index =4)

        techUser1 = st.radio(
        "What is the central disctinction between supervised and unsupervised learning",
        ["Supervised learning uses labelled datasets", 
         "Unsupervised learning may happen anytime ", 
         "Supervised learning is performed by supervised personnel", 
         "Supervised learning supersedes unsupervised learning ",
         "I don't know"], index =4)

        techUser2 = st.radio(
        "Which of the following statements is true?",
        ["Machine Learning is part of AI", 
         "Machine Learning and AI are mutually exclusive", 
         "AI and ML are the same ", 
         "AI is a part of ML",
         "I don't know"], index =4)

        techUser3 = st.radio(
        "What is a typical application of an AI at which it is usually better than non-AI",
        ["Hardware space analysis", 
         "Image recognition ", 
         "Creating annual reports", 
         "Undefined processes",
         "I don't know"], index =4)

        techUser4 = st.radio(
        "Running the same request with the same data on the same AI",
        ["increase the computing speed", 
         "never give different results", 
         "double the computing time ", 
         "could give different results",
         "I don't know"], index =4)

        techCreator1 = st.radio(
        "What always distinguishes decision trees from support vector machine?",
        ["Decision trees are trained faster", 
         "Decision trees generate more predictions ", 
         "Decision trees are more implicit", 
         "Decision trees are more interpretable ",
         "I don't know"], index =4)

        techCreator2 = st.radio(
        "What is typical split of testing and training data for development purposes",
        ["80% Training and 20% Testing", 
         "40% Training, 40% Testing, 20% Train-Testing together", 
         "95% Testing and 5% Training", 
         "It does not matter",
         "I don't know"], index =4)

        techCreator3 = st.radio(
        "What is not a strictly necessary part of a single AI system's development process?",
        ["Data preprocessing", 
         "Model definition", 
         "Benchmarking", 
         "Training/Learning",
         "I don't know"], index =4)

        techCreator4 = st.radio(
        "What is not part of an ANN?",
        ["Input layer", 
         "User layer", 
         "Output layer", 
         "Hidden layer",
         "I don't know"], index =4)


        st.markdown("**As a final evaluation, please rate the different types of explanations (0-10). This is a general grade that you you would give to the different explanation methods.**")


        shap = st.slider('SHAP', 0, 10)
        dt = st.slider('Decision tree', 0, 10)
        counterfactual  = st.slider("Counterfactual", 0, 10)
        visualmap = st.slider("Visual map", 0, 10)
        favourite = st.radio("**What was your favourite type of epxlanation?**", ('SHAP', 'Decision tree', 'Counterfactual', 'Visual map'))
        why = st.text_area('**Please explain why**', "")
        # Every form must have a submit button.

        submitted = st.form_submit_button("Submit")
        if submitted:
            st.session_state.oocsi.send('XAI_endcomparison', {#                     'participant_ID': st.session_state.participantID,
                    'gender': gender,
                    'age': age,
                    'educationLevel': educationlevel,
                    # 'socio1': socio1,
                    # 'socio2': socio2,
                    # 'socio3': socio3,
                    # 'socio4': socio4,
                    'techUser1': techUser1,
                    'techUser2': techUser2,
                    'techUser3': techUser3,
                    'techUser4': techUser4,
                    'techCreator1': techCreator1,
                    'techCreator2': techCreator2,
                    'techCreator3': techCreator3,
                    'techCreator4': techCreator4,                 
                    'participant_ID': st.session_state.participantID,
                    'shap': shap,
                    'decisiontree': dt,
                    'counterfactual': counterfactual,
                    'visualmap': visualmap,
                    'favourite': favourite,
                    'why': why                 
                    })
            st.balloons()
            switch_page('thankyou')
    # Execute your app
    # embed streamlit docs in a streamlit app

