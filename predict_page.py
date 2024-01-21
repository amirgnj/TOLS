import pickle
import numpy as np
import pandas as pd
import streamlit as st
import xgboost as xgb
import sklearn

def load_model():
    with open('saved.pkl', 'rb') as file:
        data = pickle.load(file)
    return data


data = load_model()

column_names = ['Age', 'BMI', 'CCI', 'MELD', 'Number_lesions', 'Size_lesion',
       'Platelets', 'Total_bili', 'Hemoglobin', 'Creatinine', 'ASA_1', 'ASA_2',
       'ASA_3', 'ASA_4', 'Indication_CCC', 'Indication_CRLM',
       'Indication_Donor', 'Indication_HCC', 'Indication_Benign',
       'Indication_Other malignant', 'Cirrhosis_Child A', 'Cirrhosis_Child B',
       'Cirrhosis_No', 'Approach_Hand Assisted', 'Approach_Laparoscopic',
       'Approach_Open', 'Approach_Robotic']

# Create the DataFrame
#sample2 = pd.DataFrame(columns=column_names)


# Create a row with all values equal to zero
#row_data = [0] * len(column_names)
#sample2.loc[0] = row_data

model = data["model"]
sample = data["sample"]

#sample2 = data["sample"]

def show_predict_page():
    st.title("Textbook Outcome in Liver Surgery")

    st.info("###### 📚 This risk calculator was developed and validated using data from a multi-institutional cohort of 2,059 adult patients who underwent liver surgery between 2010 and 2022. (AUC: 0.73)")
    st.info("###### 💡 Tip: The variables have set ranges determined by our dataset. Should there be a need to enter a value beyond these ranges, please use the maximum or minimum value available.")


    st.write("""### Please  provide the following information:""")
    
    Age = st.slider("Age", 18, 95, 18)
    BMI = st.slider("BMI", 10, 70, 25)
    col0 = st.columns(2)
    ASA_ = col0[0].selectbox('ASA class', ("1", "2", "3", "4"))
    Cirrhosis_ = col0[1].selectbox('Cirrhosis', ("No", "Child A", "Child B"))

    CCI = st.slider("Charlson Comorbidity Index", 0, 14, 5)
    MELD = st.slider("MELD score", 6, 28, 7)


    cols = st.columns(2)
    Number_lesions = cols[0].number_input("Number of lesions", 0, 30, 1)
    Size_lesion = cols[1].number_input("Maximum lesion size (mm)", 0, 260, 30)
  
    cols2 = st.columns(2)
    Hemoglobin = cols2[0].number_input("Hemoglobin", 7, 18, 13)
    Platelets = cols2[1].number_input("Platelets", 40, 800, 200, 10)
    cols3 = st.columns(2)
    Creatinine = cols3[0].number_input("Creatinine", 0.30, 9.00, 1.00, 0.1)
    Total_bili = cols3[1].number_input("Total bilirubin", 0.10, 15.00, 1.00, 0.1)

    col4 = st.columns(2)
    Indication_ = col4[0].selectbox('Indication', ("Benign", "HCC", "CRLM", "CCC", "Other malignant"))
    Approach_ = col4[1].selectbox('Approach', ("Open", "Laparoscopic", "Hand Assisted", "Robotic"))



    


    ok = st.button("Predict the chance of TOLS")

    if ok:
        sample["Age"] = Age
        sample["BMI"] = BMI

        sample["ASA_"+ASA_] = 1
        sample["Cirrhosis_"+Cirrhosis_] = 1

        sample["CCI"] = CCI
        sample["MELD"] = MELD
        sample["Number_lesions"] = Number_lesions
        sample["Size_lesion"] = Size_lesion
        sample["Hemoglobin"] = Hemoglobin
        sample["Platelets"] = Platelets
        sample["Creatinine"] = Creatinine
        sample["Total_bili"] = Total_bili

        sample["Indication_"+Indication_] = 1
        sample["Approach_"+Approach_] = 1




        
        chance = model.predict_proba(sample)
        #print("XGBoost version:", xgb.__version__)
        #st.subheader(sample)

        st.subheader(f"Estimated chance of TOLS: {chance[0][1]*100:.2f}%")
        #st.subheader(sklearn.__version__)


    reset = st.button("Reset")
    if reset:
        sample.loc[:,:] = 0
    

    st.error("###### ❗ Disclaimer: Please note that this tool does not reflect causal relationships between input variables and the outcome, and therefore it should not be used in isolation to dictate surgical planning.")






        

