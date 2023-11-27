import streamlit as st
import pandas as pd
import numpy as np  
import pickle
import warnings
warnings.filterwarnings('ignore')


app1_rf_pickle = open('rf_mobile_fixed_strata.pickle', 'rb')
bestRF_fetal = pickle.load(app1_rf_pickle) 
app1_rf_pickle.close()

st.title('Fetal Health Classification: A Machine Learning App')
st.image('fetal_health_image.gif', width = 650)
st.subheader('Utilize our advanced Machine Learning application to predict fetal health classifications.')
st.write('To ensure optimal results, please ensure that your data strictly adheres to the specified format outlined below:')

df_sample_data = pd.read_csv('midterm_app1_sample_data.csv', index_col = 0) 
st.dataframe(df_sample_data)

#read in user's data
fetal_health_file = st.file_uploader('Upload your data')



if fetal_health_file is None:
    st.write('')
else:
    df_user_fetal_health_file = pd.read_csv(fetal_health_file)
    #predict class/probability given user's data
    predicted_classes = bestRF_fetal.predict(df_user_fetal_health_file)
    predicted_probabilities = bestRF_fetal.predict_proba(df_user_fetal_health_file)
    highest_proba = np.max(predicted_probabilities,axis=1)

    #store predicted class/probability in the user dataframe
    df_user_fetal_health_file['Predicted Fetal Health'] = predicted_classes
    df_user_fetal_health_file['Prediction Probability (%)'] = highest_proba

    #display user's dataframe
    pd.set_option('display.precision', None)

    #highlight by class
    def highlight_class_cell(val):
        if val == 'Normal':
            color = 'lime'
        elif val == 'Suspect':
            color = 'yellow'
        elif val == 'Pathological':
            color = 'orange'
        else:
            color = ''
        return f'background-color: {color}'
        
    highlighted_user_df = df_user_fetal_health_file.style.applymap(highlight_class_cell, subset=['Predicted Fetal Health'])

    st.subheader('Predicting Fetal Health Class')
    st.dataframe(highlighted_user_df)

st.subheader("Prediction Performance")
tab1, tab2, tab3 = st.tabs(["Feature Importance", "Confusion Matrix", "Classification Report"])

with tab1:
  st.image('midterm_app1_FI_fixed_strata.svg')
with tab2:
  st.image('midterm_app1_cm_fixed_strata.svg')
with tab3:
  #read in classification report from our training dataset + rename column
  df_classifcation_report = pd.read_csv('midterm_app1_rf_class_report_fixed_strata.csv', index_col = 0)
  st.dataframe(df_classifcation_report)

