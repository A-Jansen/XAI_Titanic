"""
These are the meta data and instructions
Author: Anniek Jansen
Course: Human AI interaction

Instructions:
(For Anniek: conda activate py (activate local environment where the right packages are installed))
1. Pip install streamlit, oocsi, streamlit_extras in your python environment
2. Save this file somewhere on your computer
3. In the command line, cd to where your file is: "cd/...../folder

4. To run it: streamlit run app.py 
5. Click on the link it provides you
6. You need to click sometimes rerun in the website


To hide menu: copy paste this in config.toml
[ui]
hideSidebarNav = true


 task: try to remove the local profil index variable for each page associated with the indices realted to them (SHPA, tree, visual, and counterfactual)

"""
import datetime
from datetime import datetime
import streamlit as st
import pandas as pd
# from oocsi import __init__ as OOCSI
from oocsi_source import OOCSI
from uuid import uuid4
from streamlit_extras.switch_page_button import switch_page
import requests
import json


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


st.session_state.current_page_title = "Introduction"
page_start_time = None
record_page_start_time()

# Check if 'participantID' already exists in session_state
# If not, then initialize it
if 'participantID' not in st.session_state:
    st.session_state.participantID = "P" + \
        uuid4().__str__().replace('-', '')[0:10]
    st.session_state.pages = [
        'SHAP', 'DecisionTree', 'counterfactual', 'visualMap']
    # st.session_state.profileIndices = [25, 112, 56, 22]
    st.session_state.profileIndices_SHAP = [28, 36, 48, 21]#28, 110, 50, 21  ##another women --> 48
    st.session_state.profileIndices_counter = [26, 113, 57, 69]
    st.session_state.profileIndices_Tree = [27, 114, 58, 24]
    st.session_state. profileIndices_visual = [25, 112, 56, 22]

if 'oocsi' not in st.session_state:
    st.session_state.oocsi = OOCSI('', 'oocsi.id.tue.nl')


# st.write(st.session_state.participantID)

st.set_page_config(page_title="XAI research", layout="wide")

header1, header2, header3 = st.columns([1, 2, 1])
consent_form1, consent_form2, consent_form3 = st.columns([1, 4, 1])

# with header2:
# st.title("Consent form")
# st.write("For debugging:")
# st.write(st.session_state.participantID)

with consent_form2:
    st.header('Information form for participants')
    st.write(
        '''Hello and thank you for taking the time to participate in this survey.''')
    st.write('''This document gives you information about the study Comparing Explainable AI (XAI) methods. Before the study begins, it is important that you learn about the procedure followed in this study and that you give your informed consent for voluntary participation. Please read this document carefully.  ''')

    st.subheader('Aim and benefit of the study')
    st.write('''The aim of this study is to measure the satisfaction of users with different types of XAI methods. 
    This information is used to have better understandable/ more satisfying type of explanations in future applications.  ''')
    st.write('''This study is performed by Rachel Wang, François Leborgne and Anniek Jansen, all EngD trainees of the Designing Human-System Interaction program and for this study under the supervision of Chao Zhang of the Human-Technology Interaction group.''')

    st.subheader('Procedure')
    st.markdown('''During this project we ask you to: 
-	Look at different predictions from an AI model (predicting the survival of passengers of the Titanic)
-	Complete a short survey (9 questions) for each of the four types of explanation method
-	Complete a survey at the end of the study with demographic information
-	Complete a second survey at the end to compare the explanation methods and explain why certain methods had your preference.
''')

    st.subheader('Risks')
    st.markdown(
        "The study does not involve any risks, detrimental side effects, or cause discomfort.")

    st.subheader("Duration")
    st.markdown(
        "The instructions, measurements and debriefing will take approximately 30 minutes.")

    st.subheader("Voluntary")
    st.markdown('''Your participation is completely voluntary. You can refuse to participate without giving any reasons and you can stop your participation at any time during the study. You can also withdraw your permission to use your data immediately after completing the study. None of this will have any negative consequences for you whatsoever.''')

    st.subheader("Confidentiality and use, storage, and sharing of data")
    st.markdown('''
     All research conducted at the Human-Technology Interaction Group adheres to the Code of Ethics of the NIP (Nederlands Instituut voor Psychologen – Dutch Institute for Psychologists), and this study has been approved by the Ethical Review Board of the department. 

    In this study demographic data (gender, age, education level, highest level of education, data literacy, AI expertise), and experimental data (response to questionnaires and duration of experiment) will be recorded, analyzed, and stored. 
    The goal of collecting, analyzing, and storing this data is to answer the research question and publish the results in the scientific literature. 
    Next to the research data, we ask you to leave your email address if you are willing to participate in a follow-up interview and we might need to store your name and bank account in order to compensate you for your participation. 
    This data will be stored separately from the research data and deleted after the interview/ payment.
                To protect your privacy, all data that can be used to personally identify you will be stored on an encrypted server of the Human Technology Interaction group for at least 10 years that is only accessible by selected HTI staff members. No information that can be used to personally identify you will be shared with others. 

    During the study, the data will be stored on encrypted laptops and DataFoundry  - a platform developed by the Department of Industrial Design at TU/e and is GDPR compliant. After the analyses, the data will also be made available on OSF (open science framework, a place to share research data open source).  

    The data collected in this study might also be of relevance for future research projects within the Human Technology Interaction group as well as for other researchers. The aim of those studies might be unrelated to the goals of this study. 
    The collected data will therefore also be made available to the general public in an online data repository.
    The coded data collected in this study and that will be released to the public will (to the best of our knowledge and ability) not contain information that can identify you. It will include all answers you provide during the study, including demographic variables (e.g., age and gender) if you choose to provide these during the study. 

    At the bottom of this consent form, you can indicate whether or not you agree with participation in this study. You can also indicate whether you agree with the distribution of your data by means of a secured online data repository with open access for the general public and the distribution of your data by means of a secured online data repository with open access for the general public. You are not obliged to let us use and share your data. If you are not willing to share your data in this way, you can still participate in this study. Your data will be used in the scientific article but not shared with other researchers.

    No video or audio recordings are made that could identify you.
    
    ''')

    st.subheader("Further information")
    st.markdown('''If you want more information about this study, the study design, or the results, you can contact Anniek Jansen (contact email: a.jansen@tue.nl ). 
    If you have any complaints about this study, please contact the supervisor, Chao Zhang (C.Zhang.5@tue.nl)  You can report irregularities related to scientific integrity to confidential advisors of the TU/e.
    ''')

    st.subheader("Informed consent form")
    st.markdown('''
    -   I am 18 years or older
    -	I have read and understood the information of the corresponding information form for participants. 
    -	I have been given the opportunity to ask questions. My questions are sufficiently answered, and I had sufficient time to decide whether I participate. 
    -	I know that my participation is completely voluntary. I know that I can refuse to participate and that I can stop my participation at any time during the study, without giving any reasons. I know that I can withdraw permission to use my data directly after the experiment.
    -	I agree to voluntarily participate in this study carried out by the research group Human Technology Interaction and Industrial Design of the Eindhoven University of Technology.
    -	I know that no information that can be used to personally identify me or my responses in this study will be shared with anyone outside of the research team.
    ''')

    prolificID = st.text_input(
        "Please enter/paste here your unique prolific ID")
    st.session_state.participantID = prolificID

    OSF = st.radio("I ... (please select below) give permission to make my anonymized recorded data available to others in a public online data repository, and allow others to use this data for future research projects unrelated to this study.",
                   ('do',
                    'do not'), index=1)

    st.subheader("Consent")
    agree = st.radio(
        'I consent to processing my personal data gathered during the research in the way described in the information sheet.',
        ('do',
        'do not'), index=1)

    consentforOSF = ""
    if OSF == 'do':
        consentforOSF = 'yes'
    else:
        consentforOSF = 'no'
        


    if agree == "do":
        if len(prolificID) >0:
            st.write(
                'Thank you! Please continue to the next page to start the experiment')
            if st.button("Next page"):
                if page_start_time:
                    record_page_duration_and_send()
                # record_page_start_time()
                st.session_state.oocsi.send('XAI_consent', {
                    'participant_ID': st.session_state.participantID,
                    'expert': "yes",
                    'consent': 'yes',
                    'consentForOSF': consentforOSF
                })
                switch_page("explanationpage")
        else:
            st.warning('Please add your prolific ID')
    else:
        if st.button("Next page"):
            st.session_state.oocsi.send('XAI_consent', {
                'participant_ID': st.session_state.participantID,
                'expert': "yes",
                'consent': 'no',
                'consentForOSF': consentforOSF
            })
            switch_page('noconsent')
