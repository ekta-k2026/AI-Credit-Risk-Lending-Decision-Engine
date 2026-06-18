import streamlit as st
import joblib
import pandas as pd
model = joblib.load(
    "models/xgboost_model.pkl"
)
st.success("XGBoost Model Loaded Successfully")

st.title("AI Credit Risk Lending Decision Engine")

st.subheader("Borrower Information")

loan_amnt = st.number_input(
    "Loan Amount",
    value=10000
)

annual_inc = st.number_input(
    "Annual Income",
    value=50000
)

dti = st.number_input(
    "Debt-to-Income Ratio",
    value=15.0
)

int_rate = st.number_input(
    "Interest Rate",
    value=10.0
)

installment = st.number_input(
    "Monthly Installment",
    value=300.0
)

avg_fico = st.number_input(
    "FICO Score",
    value=700
)

emp_length_num = st.number_input(
    "Employment Length (Years)",
    value=5
)
if st.button("Predict Risk"):

    input_df = pd.DataFrame({
        "loan_amnt": [loan_amnt],
        "annual_inc": [annual_inc],
        "dti": [dti],
        "int_rate": [int_rate],
        "installment": [installment],
        "avg_fico": [avg_fico],
        "emp_length_num": [emp_length_num],

        "home_ownership_MORTGAGE": [0],
        "home_ownership_OWN": [0],
        "home_ownership_RENT": [1],

        "purpose_credit_card": [0],
        "purpose_debt_consolidation": [0],
        "purpose_home_improvement": [0],
        "purpose_house": [0],
        "purpose_major_purchase": [0],
        "purpose_medical": [0],
        "purpose_moving": [0],
        "purpose_other": [1],
        "purpose_renewable_energy": [0],
        "purpose_small_business": [0],
        "purpose_vacation": [0],

        "grade_B": [0],
        "grade_C": [0],
        "grade_D": [0],
        "grade_E": [0],
        "grade_F": [0],
        "grade_G": [0]
    })

    probability = model.predict_proba(
        input_df
    )[0][1]
    risk_score = round(probability * 100)
    if risk_score < 30:
        risk_category = "Low Risk"
    elif risk_score < 60:
        risk_category = "Medium Risk"
    else:
        risk_category = "High Risk"
    if risk_category == "Low Risk":
        decision = "Approve"
    elif risk_category == "Medium Risk":
        decision = "Review"
    else:
        decision = "Reject"
    st.subheader("Risk Assessment")
    
    st.write(f"Default Probability: {probability:.2%}")
    st.write(f"Risk Score: {risk_score}")
    st.write(f"Risk Category: {risk_category}")
    if decision == "Approve":
        st.success(f"Lending Decision: {decision}")

    elif decision == "Review":
        st.warning(f"Lending Decision: {decision}")

    else:
        st.error(f"Lending Decision: {decision}")

    


