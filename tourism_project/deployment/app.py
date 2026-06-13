
import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# --------------------------------------------------
# Load Model
# --------------------------------------------------

model_path = hf_hub_download(
    repo_id="NiteshKK/visit-with-us-sales-prediction",
    repo_type="dataset",
    filename="best_tourism_sales_model.joblib"
)

model = joblib.load(model_path)

# --------------------------------------------------
# Streamlit UI
# --------------------------------------------------

st.set_page_config(
    page_title="Visit With Us Sales Prediction",
    layout="wide"
)

st.title("Visit With Us Sales Prediction")

st.write(
"""
Predict whether a customer is likely to purchase a tourism package
based on demographic information and sales interaction details.
"""
)

# --------------------------------------------------
# User Inputs
# --------------------------------------------------

col1, col2 = st.columns(2)

with col1:

    Age = st.number_input(
        "Age",
        min_value=18,
        max_value=80,
        value=35
    )

    TypeofContact = st.selectbox(
        "Type Of Contact",
        [
            "Self Enquiry",
            "Company Invited"
        ]
    )

    CityTier = st.selectbox(
        "City Tier",
        [1, 2, 3]
    )

    DurationOfPitch = st.number_input(
        "Duration Of Pitch",
        min_value=1,
        max_value=100,
        value=15
    )

    Occupation = st.selectbox(
        "Occupation",
        [
            "Salaried",
            "Small Business",
            "Large Business",
            "Free Lancer"
        ]
    )

    Gender = st.selectbox(
        "Gender",
        [
            "Male",
            "Female"
        ]
    )

    NumberOfPersonVisiting = st.number_input(
        "Number Of Persons Visiting",
        min_value=1,
        max_value=10,
        value=2
    )

    NumberOfFollowups = st.number_input(
        "Number Of Followups",
        min_value=0,
        max_value=10,
        value=2
    )

with col2:

    ProductPitched = st.selectbox(
        "Product Pitched",
        [
            "Basic",
            "Standard",
            "Deluxe",
            "Super Deluxe",
            "King"
        ]
    )

    PreferredPropertyStar = st.selectbox(
        "Preferred Property Star",
        [1, 2, 3, 4, 5]
    )

    MaritalStatus = st.selectbox(
        "Marital Status",
        [
            "Single",
            "Married",
            "Divorced",
            "Unmarried"
        ]
    )

    NumberOfTrips = st.number_input(
        "Number Of Trips",
        min_value=0,
        max_value=30,
        value=3
    )

    Passport = st.selectbox(
        "Passport",
        [0, 1]
    )

    PitchSatisfactionScore = st.selectbox(
        "Pitch Satisfaction Score",
        [1, 2, 3, 4, 5]
    )

    OwnCar = st.selectbox(
        "Own Car",
        [0, 1]
    )

    NumberOfChildrenVisiting = st.number_input(
        "Number Of Children Visiting",
        min_value=0,
        max_value=10,
        value=1
    )

    Designation = st.selectbox(
        "Designation",
        [
            "Executive",
            "Manager",
            "Senior Manager",
            "AVP",
            "VP"
        ]
    )

    MonthlyIncome = st.number_input(
        "Monthly Income",
        min_value=10000,
        max_value=100000,
        value=25000
    )

# --------------------------------------------------
# Prediction Input
# --------------------------------------------------

input_data = pd.DataFrame([{
    "Age": Age,
    "TypeofContact": TypeofContact,
    "CityTier": CityTier,
    "DurationOfPitch": DurationOfPitch,
    "Occupation": Occupation,
    "Gender": Gender,
    "NumberOfPersonVisiting": NumberOfPersonVisiting,
    "NumberOfFollowups": NumberOfFollowups,
    "ProductPitched": ProductPitched,
    "PreferredPropertyStar": PreferredPropertyStar,
    "MaritalStatus": MaritalStatus,
    "NumberOfTrips": NumberOfTrips,
    "Passport": Passport,
    "PitchSatisfactionScore": PitchSatisfactionScore,
    "OwnCar": OwnCar,
    "NumberOfChildrenVisiting": NumberOfChildrenVisiting,
    "Designation": Designation,
    "MonthlyIncome": MonthlyIncome
}])

# --------------------------------------------------
# Prediction
# --------------------------------------------------

if st.button("Predict Purchase"):

    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(input_data)[0][1]

    st.subheader("Prediction Result")

    if prediction == 1:

        st.success(
            f"Customer is likely to purchase the package.\n\n"
            f"Probability: {probability:.2%}"
        )

    else:

        st.error(
            f"Customer is unlikely to purchase the package.\n\n"
            f"Probability: {probability:.2%}"
        )
