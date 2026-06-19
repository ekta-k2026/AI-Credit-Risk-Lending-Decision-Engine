import streamlit as st
import joblib
import pandas as pd

# Load Model

model = joblib.load(
"models/xgboost_model.pkl"
)

# App Title

st.title("🏦 AI Credit Risk Lending Decision Engine")

st.markdown(
"Assess borrower risk and generate lending decisions using an AI-powered credit risk model."
)

# Borrower Information

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

home_ownership = st.selectbox(
"Home Ownership",
["RENT", "OWN", "MORTGAGE"]
)

purpose = st.selectbox(
"Loan Purpose",
[
"credit_card",
"debt_consolidation",
"home_improvement",
"house",
"major_purchase",
"medical",
"moving",
"other",
"renewable_energy",
"small_business",
"vacation"
]
)

grade = st.selectbox(
"Loan Grade",
["A", "B", "C", "D", "E", "F", "G"]
)

# Prediction

if st.button("Predict Risk"):

 
    input_df = pd.DataFrame({
    "loan_amnt": [loan_amnt],
    "annual_inc": [annual_inc],
    "dti": [dti],
    "int_rate": [int_rate],
    "installment": [installment],
    "avg_fico": [avg_fico],
    "emp_length_num": [emp_length_num]
})

# Home Ownership Encoding
    input_df["home_ownership_MORTGAGE"] = int(
    home_ownership == "MORTGAGE"
)

    input_df["home_ownership_OWN"] = int(
    home_ownership == "OWN"
)

    input_df["home_ownership_RENT"] = int(
    home_ownership == "RENT"
)

# Purpose Encoding
    for p in [
        "credit_card",
        "debt_consolidation",
        "home_improvement",
        "house",
        "major_purchase",
        "medical",
        "moving",
        "other",
        "renewable_energy",
        "small_business",
        "vacation"
]:
        input_df[f"purpose_{p}"] = int(
        purpose == p
    )

# Grade Encoding
    for g in ["B", "C", "D", "E", "F", "G"]:
        input_df[f"grade_{g}"] = int(
        grade == g
    )

# Prediction
    probability = model.predict_proba(
        input_df
    )[0][1]

    risk_score = round(
     probability * 100
    )

# Risk Category
    if risk_score < 30:
        risk_category = "Low Risk"
    elif risk_score < 60:
        risk_category = "Medium Risk"
    else:
        risk_category = "High Risk"

# Lending Decision
    if risk_category == "Low Risk":
        decision = "Approve"
    elif risk_category == "Medium Risk":
        decision = "Review"
    else:
        decision = "Reject"

# Results
    st.subheader("📊 Risk Assessment")

    st.metric(
    "Default Probability",
    f"{probability:.2%}"
    )

    st.metric(
      "Risk Score",
       risk_score
)

    st.write(
    f"Risk Category: **{risk_category}**"
    )

    if decision == "Approve":
        st.success(
        f"Lending Decision: {decision}"
    )

    elif decision == "Review":
        st.warning(
        f"Lending Decision: {decision}"
    )

    else:
        st.error(
        f"Lending Decision: {decision}"
    )

    st.subheader("📋 Decision Explanation")

    if decision == "Approve":
      st.write(
        "The borrower demonstrates a relatively low risk profile based on the information provided."
    )

    elif decision == "Review":
      st.write(
        "The borrower presents moderate risk. Additional review is recommended before approval."
    )

    else:
        st.write(
        "The borrower presents elevated risk. Loan approval is not recommended."
    )


st.markdown("---")

st.caption(
"Built with XGBoost, Streamlit, and Explainable AI techniques."
)
