import streamlit as st
import joblib
import pandas as pd

# Load Model

model = joblib.load(
"models/xgboost_model.pkl"
)

# App Title

st.title("🏦 AI Credit Risk Assessment")

st.markdown(
"Assess borrower risk and generate lending decisions using an AI-powered credit risk model."
)

# Borrower Information

st.subheader("Borrower Information")

col1, col2 = st.columns(2)

with col1:

    loan_amnt = st.number_input(
        "Loan Amount",
        value=10000
    )

    annual_inc = st.number_input(
        "Annual Income",
        value=50000
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

with col2:

    dti = st.number_input(
        "Debt-to-Income Ratio",
        value=15.0
    )

    int_rate = st.number_input(
        "Interest Rate",
        value=10.0
    )

    loan_tenure = st.number_input(
        "Loan Tenure (Years)",
        value=5,
        min_value=1
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
            "vacation",
            "education",
            "vehicle"
        ]
    )

    grade = st.selectbox(
        "Loan Grade",
        ["A", "B", "C", "D", "E", "F", "G"]
    )
monthly_rate = (int_rate / 100) / 12
num_payments = loan_tenure * 12

if monthly_rate == 0:
    installment = loan_amnt / num_payments
else:
    installment = (
        loan_amnt
        * monthly_rate
        * (1 + monthly_rate) ** num_payments
    ) / (
        (1 + monthly_rate) ** num_payments - 1
    )

st.subheader("💰 Loan Summary")    

st.metric(
    "Calculated Monthly Installment",
    f"₹{installment:.2f}"
)


# Prediction

left, center, right = st.columns([1, 2, 1])


with center:
    predict = st.button(
        "🔍 Predict Risk",
        use_container_width=True
    )


if predict:



 
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
    if purpose == "education":
        purpose_for_model = "other"

    elif purpose == "vehicle":
        purpose_for_model = "major_purchase"

    else:
        purpose_for_model = purpose

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
           purpose_for_model == p
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

    elif risk_score < 50:
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
        "The borrower has a low predicted probability of default. "
        "The combination of credit profile, income level, loan characteristics, "
        "and repayment capacity indicates relatively low lending risk."
    )

    elif decision == "Review":
     st.write(
        "The borrower presents moderate risk. Additional verification or manual review "
        "is recommended before making a final lending decision."
    )
    else:
     st.write(
        "The model predicts a high probability of default. "
        "The borrower's financial profile and loan characteristics indicate "
        "elevated lending risk."
    )
    risk_factors = []

    if avg_fico < 600:
      risk_factors.append("Low FICO score")

    if annual_inc < 30000:
      risk_factors.append("Low annual income")

    if loan_amnt > annual_inc:
      risk_factors.append("Loan amount exceeds annual income")

    if int_rate > 20:
      risk_factors.append("High interest rate")

    if emp_length_num < 2:
      risk_factors.append("Short employment history")

    if risk_factors:
      st.markdown("### Key Risk Factors")
      for factor in risk_factors:
        st.write(f"• {factor}")

st.markdown("---")

st.caption(
"Built with XGBoost, Streamlit, and Explainable AI techniques."
)
