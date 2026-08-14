# import streamlit as st
# import pandas as pd
# import joblib
# from pathlib import Path


# # ============================================================
# # PAGE CONFIGURATION
# # ============================================================

# st.set_page_config(
#     page_title="Credit Card Fraud Detection",
#     page_icon="💳",
#     layout="wide"
# )


# # ============================================================
# # PATHS
# # ============================================================

# BASE_DIR = Path(__file__).resolve().parent
# MODEL_DIR = BASE_DIR / "models"


# # ============================================================
# # LOAD MODEL
# # ============================================================

# @st.cache_resource
# def load_model():

#     model = joblib.load(
#         MODEL_DIR / "fraud_detection_random_forest.pkl"
#     )

#     threshold = joblib.load(
#         MODEL_DIR / "threshold.pkl"
#     )

#     return model, threshold


# model, threshold = load_model()


# # ============================================================
# # FEATURES
# # ============================================================

# FEATURES = [
#     "Time",
#     "V1", "V2", "V3", "V4", "V5", "V6", "V7",
#     "V8", "V9", "V10", "V11", "V12", "V13", "V14",
#     "V15", "V16", "V17", "V18", "V19", "V20", "V21",
#     "V22", "V23", "V24", "V25", "V26", "V27", "V28",
#     "Amount"
# ]


# # ============================================================
# # HEADER
# # ============================================================

# st.title("💳 Credit Card Fraud Detection System")

# st.markdown(
#     """
#     This application uses a **Random Forest Machine Learning model**
#     to detect potentially fraudulent credit card transactions.
#     """
# )

# st.divider()


# # ============================================================
# # MODEL INFORMATION
# # ============================================================

# col1, col2, col3 = st.columns(3)

# with col1:
#     st.metric("Model", "Random Forest")

# with col2:
#     st.metric("Decision Threshold", f"{threshold:.2f}")

# with col3:
#     st.metric("F1 Score", "83.70%")


# st.divider()


# # ============================================================
# # INPUT SECTION
# # ============================================================

# st.subheader("Transaction Details")

# st.info(
#     "Enter the transaction features below. "
#     "The model will calculate the probability of fraud."
# )


# # Create input dictionary
# input_data = {}


# # Time and Amount
# col1, col2 = st.columns(2)

# with col1:
#     input_data["Time"] = st.number_input(
#         "Time",
#         min_value=0.0,
#         value=0.0
#     )

# with col2:
#     input_data["Amount"] = st.number_input(
#         "Transaction Amount",
#         min_value=0.0,
#         value=100.0
#     )


# st.subheader("Transaction Features")

# # V1-V14
# cols = st.columns(3)

# for i, feature in enumerate(FEATURES[1:15]):

#     with cols[i % 3]:

#         input_data[feature] = st.number_input(
#             feature,
#             value=0.0,
#             format="%.6f"
#         )


# # V15-V28
# cols = st.columns(3)

# for i, feature in enumerate(FEATURES[15:]):

#     with cols[i % 3]:

#         input_data[feature] = st.number_input(
#             feature,
#             value=0.0,
#             format="%.6f"
#         )


# st.divider()


# # ============================================================
# # PREDICTION
# # ============================================================

# if st.button(
#     "🔍 Analyze Transaction",
#     use_container_width=True
# ):

#     transaction = pd.DataFrame(
#         [input_data],
#         columns=FEATURES
#     )

#     # Fraud probability
#     fraud_probability = model.predict_proba(
#         transaction
#     )[0][1]

#     # Apply selected threshold
#     prediction = int(
#         fraud_probability >= threshold
#     )

#     st.subheader("Detection Result")

#     col1, col2 = st.columns(2)

#     with col1:

#         st.metric(
#             "Fraud Probability",
#             f"{fraud_probability * 100:.2f}%"
#         )

#     with col2:

#         if prediction == 1:

#             st.error(
#                 "🚨 Potential Fraudulent Transaction"
#             )

#         else:

#             st.success(
#                 "✅ Normal Transaction"
#             )


#     # Probability bar
#     st.progress(
#         min(float(fraud_probability), 1.0)
#     )

#     if prediction == 1:

#         st.warning(
#             f"The fraud probability is above the "
#             f"selected threshold of {threshold:.2f}."
#         )

#     else:

#         st.info(
#             f"The fraud probability is below the "
#             f"selected threshold of {threshold:.2f}."
#         )









import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)

# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR

# ============================================================
# FEATURES
# ============================================================

FEATURES = [
    "Time",
    "V1", "V2", "V3", "V4", "V5", "V6", "V7",
    "V8", "V9", "V10", "V11", "V12", "V13", "V14",
    "V15", "V16", "V17", "V18", "V19", "V20", "V21",
    "V22", "V23", "V24", "V25", "V26", "V27",
    "V28", "Amount"
]

# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    model_path = MODEL_DIR / "fraud_detection_random_forest.pkl"
    threshold_path = MODEL_DIR / "threshold.pkl"

    if not model_path.exists():
        st.error(
            "❌ fraud_detection_random_forest.pkl was not found. "
            "Make sure the model file is in the same folder as app.py."
        )
        st.stop()

    if not threshold_path.exists():
        st.error(
            "❌ threshold.pkl was not found. "
            "Make sure the threshold file is in the same folder as app.py."
        )
        st.stop()

    model = joblib.load(model_path)
    threshold = joblib.load(threshold_path)

    return model, threshold


model, threshold = load_model()

# ============================================================
# HEADER
# ============================================================

st.title("💳 Credit Card Fraud Detection System")

st.markdown(
    """
This application uses a **Random Forest Machine Learning model**
to detect potentially fraudulent credit card transactions.
"""
)

st.divider()

# ============================================================
# MODEL INFORMATION
# ============================================================

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "Model",
        "Random Forest"
    )

with col2:
    st.metric(
        "Decision Threshold",
        f"{threshold:.4f}"
    )

st.divider()

# ============================================================
# TRANSACTION DETAILS
# ============================================================

st.subheader("💳 Transaction Details")

st.info(
    "Enter the transaction values below and click "
    "**Analyze Transaction** to get the fraud probability."
)

input_data = {}

# ============================================================
# TIME AND AMOUNT
# ============================================================

st.subheader("Transaction Information")

col1, col2 = st.columns(2)

with col1:
    input_data["Time"] = st.number_input(
        "Time",
        min_value=0.0,
        value=0.0,
        format="%.6f"
    )

with col2:
    input_data["Amount"] = st.number_input(
        "Transaction Amount",
        min_value=0.0,
        value=0.0,
        format="%.6f"
    )

# ============================================================
# V1 - V14
# ============================================================

st.subheader("Transaction Features V1 - V14")

cols = st.columns(3)

for i, feature in enumerate(FEATURES[1:15]):

    with cols[i % 3]:

        input_data[feature] = st.number_input(
            feature,
            value=0.0,
            format="%.6f",
            key=f"{feature}_input"
        )

# ============================================================
# V15 - V28
# ============================================================

st.subheader("Transaction Features V15 - V28")

cols = st.columns(3)

for i, feature in enumerate(FEATURES[15:29]):

    with cols[i % 3]:

        input_data[feature] = st.number_input(
            feature,
            value=0.0,
            format="%.6f",
            key=f"{feature}_input"
        )

# ============================================================
# PREDICTION
# ============================================================

st.divider()

if st.button(
    "🔍 Analyze Transaction",
    use_container_width=True
):

    # --------------------------------------------------------
    # CREATE TRANSACTION DATAFRAME
    # --------------------------------------------------------

    transaction = pd.DataFrame(
        [input_data],
        columns=FEATURES
    )

    # --------------------------------------------------------
    # FRAUD PROBABILITY
    # --------------------------------------------------------

    probabilities = model.predict_proba(transaction)[0]

    if hasattr(model, "classes_"):

        classes = list(model.classes_)

        if 1 in classes:
            fraud_index = classes.index(1)
            fraud_probability = float(
                probabilities[fraud_index]
            )
        else:
            fraud_probability = 0.0

    else:
        fraud_probability = float(probabilities[1])

    # --------------------------------------------------------
    # APPLY DECISION THRESHOLD
    # --------------------------------------------------------

    prediction = int(
        fraud_probability >= threshold
    )

    # ========================================================
    # DETECTION RESULT
    # ========================================================

    st.subheader("🎯 Detection Result")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Fraud Probability",
            f"{fraud_probability * 100:.2f}%"
        )

    with col2:
        st.metric(
            "Decision Threshold",
            f"{threshold * 100:.2f}%"
        )

    with col3:

        if prediction == 1:
            st.error("🚨 Predicted: FRAUD")
        else:
            st.success("✅ Predicted: NORMAL")

    # ========================================================
    # PROBABILITY BAR
    # ========================================================

    st.progress(
        min(
            max(fraud_probability, 0.0),
            1.0
        )
    )

    # ========================================================
    # FINAL EXPLANATION
    # ========================================================

    st.divider()

    if prediction == 1:

        st.error(
            f"🚨 Potential fraudulent transaction detected. "
            f"The fraud probability is "
            f"**{fraud_probability * 100:.2f}%**, which is "
            f"greater than or equal to the decision threshold "
            f"of **{threshold * 100:.2f}%**."
        )

    else:

        st.success(
            f"✅ Transaction classified as normal. "
            f"The fraud probability is "
            f"**{fraud_probability * 100:.2f}%**, which is "
            f"below the decision threshold of "
            f"**{threshold * 100:.2f}%**."
        )

    # ========================================================
    # INPUT SUMMARY
    # ========================================================

    with st.expander("📄 View Transaction Values"):

        st.dataframe(
            transaction,
            use_container_width=True
        )
