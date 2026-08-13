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
DATASET_PATH = BASE_DIR / "creditcard.csv"

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

    model = joblib.load(
        MODEL_DIR / "fraud_detection_random_forest.pkl"
    )

    threshold = joblib.load(
        MODEL_DIR / "threshold.pkl"
    )

    return model, threshold


model, threshold = load_model()

# ============================================================
# LOAD DATASET
# ============================================================

@st.cache_data
def load_dataset():

    if not DATASET_PATH.exists():
        return None

    df = pd.read_csv(DATASET_PATH)

    return df


df = load_dataset()

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
# CHECK DATASET
# ============================================================

if df is None:

    st.error(
        "❌ creditcard.csv not found. "
        "Please place creditcard.csv in the same folder as app.py."
    )

    st.stop()

# Check required columns

required_columns = FEATURES + ["Class"]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:

    st.error(
        f"❌ Missing columns in dataset: {missing_columns}"
    )

    st.stop()

# ============================================================
# DATASET INFORMATION
# ============================================================

normal_count = int((df["Class"] == 0).sum())
fraud_count = int((df["Class"] == 1).sum())

# ============================================================
# MODEL INFORMATION
# ============================================================

col1, col2, col3, col4 = st.columns(4)

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

with col3:
    st.metric(
        "Normal Transactions",
        f"{normal_count:,}"
    )

with col4:
    st.metric(
        "Fraud Transactions",
        f"{fraud_count:,}"
    )

st.divider()

# ============================================================
# SESSION STATE
# ============================================================

if "selected_transaction" not in st.session_state:
    st.session_state.selected_transaction = None

if "actual_label" not in st.session_state:
    st.session_state.actual_label = None

# ============================================================
# TRANSACTION LOADER
# ============================================================

st.subheader("📂 Test Actual Dataset Transaction")

st.info(
    "Instead of manually entering 0.000000 values, "
    "load an actual transaction directly from creditcard.csv."
)

col1, col2 = st.columns(2)

with col1:

    if st.button(
        "🟢 Load Normal Transaction",
        use_container_width=True
    ):

        normal_rows = df[df["Class"] == 0]

        selected = normal_rows.sample(
            n=1,
            random_state=None
        ).iloc[0]

        st.session_state.selected_transaction = selected
        st.session_state.actual_label = 0

        st.rerun()

with col2:

    if st.button(
        "🔴 Load Fraud Transaction",
        use_container_width=True
    ):

        fraud_rows = df[df["Class"] == 1]

        selected = fraud_rows.sample(
            n=1,
            random_state=None
        ).iloc[0]

        st.session_state.selected_transaction = selected
        st.session_state.actual_label = 1

        st.rerun()

# ============================================================
# INPUT SECTION
# ============================================================

st.divider()

st.subheader("Transaction Details")

# ============================================================
# CURRENT TRANSACTION
# ============================================================

selected_transaction = st.session_state.selected_transaction

input_data = {}

# ------------------------------------------------------------
# If dataset transaction is loaded
# ------------------------------------------------------------

if selected_transaction is not None:

    st.success(
        "✅ Actual transaction loaded from creditcard.csv"
    )

    actual_label = int(st.session_state.actual_label)

    if actual_label == 0:
        st.info("Actual Label: NORMAL (0)")
    else:
        st.warning("Actual Label: FRAUD (1)")

    # --------------------------------------------------------
    # Time and Amount
    # --------------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        input_data["Time"] = st.number_input(
            "Time",
            value=float(selected_transaction["Time"]),
            format="%.6f",
            key="time_input"
        )

    with col2:

        input_data["Amount"] = st.number_input(
            "Transaction Amount",
            min_value=0.0,
            value=float(selected_transaction["Amount"]),
            format="%.6f",
            key="amount_input"
        )

    st.subheader("Transaction Features")

    # --------------------------------------------------------
    # V1 - V14
    # --------------------------------------------------------

    cols = st.columns(3)

    for i, feature in enumerate(FEATURES[1:15]):

        with cols[i % 3]:

            input_data[feature] = st.number_input(
                feature,
                value=float(selected_transaction[feature]),
                format="%.6f",
                key=f"{feature}_input"
            )

    # --------------------------------------------------------
    # V15 - V28
    # --------------------------------------------------------

    cols = st.columns(3)

    for i, feature in enumerate(FEATURES[15:29]):

        with cols[i % 3]:

            input_data[feature] = st.number_input(
                feature,
                value=float(selected_transaction[feature]),
                format="%.6f",
                key=f"{feature}_input"
            )

else:

    st.warning(
        "⚠️ Please click "
        "**Load Normal Transaction** or "
        "**Load Fraud Transaction** first."
    )

# ============================================================
# PREDICTION
# ============================================================

if selected_transaction is not None:

    st.divider()

    if st.button(
        "🔍 Analyze Transaction",
        use_container_width=True
    ):

        transaction = pd.DataFrame(
            [input_data],
            columns=FEATURES
        )

        # ----------------------------------------------------
        # Fraud Probability
        # ----------------------------------------------------

        probabilities = model.predict_proba(
            transaction
        )[0]

        # Safely find probability for class 1
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

            fraud_probability = float(
                probabilities[1]
            )

        # ----------------------------------------------------
        # Prediction using threshold
        # ----------------------------------------------------

        prediction = int(
            fraud_probability >= threshold
        )

        actual_label = int(
            st.session_state.actual_label
        )

        # ====================================================
        # DETECTION RESULT
        # ====================================================

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

                st.error(
                    "🚨 Predicted: FRAUD"
                )

            else:

                st.success(
                    "✅ Predicted: NORMAL"
                )

        st.progress(
            min(max(fraud_probability, 0.0), 1.0)
        )

        # ====================================================
        # ACTUAL VS PREDICTED
        # ====================================================

        st.divider()

        st.subheader("📊 Actual vs Predicted")

        col1, col2 = st.columns(2)

        with col1:

            if actual_label == 1:

                st.error(
                    "Actual Label: 🔴 FRAUD (1)"
                )

            else:

                st.success(
                    "Actual Label: 🟢 NORMAL (0)"
                )

        with col2:

            if prediction == 1:

                st.error(
                    "Predicted Label: 🔴 FRAUD (1)"
                )

            else:

                st.success(
                    "Predicted Label: 🟢 NORMAL (0)"
                )

        # ====================================================
        # FINAL VERDICT
        # ====================================================

        st.divider()

        if actual_label == prediction:

            st.success(
                "✅ Model Prediction is CORRECT"
            )

        else:

            st.warning(
                "⚠️ Model Prediction is INCORRECT"
            )

        # ====================================================
        # EXPLANATION
        # ====================================================

        if prediction == 1:

            st.warning(
                f"The fraud probability "
                f"({fraud_probability:.4f}) is greater than "
                f"or equal to the decision threshold "
                f"({threshold:.4f})."
            )

        else:

            st.info(
                f"The fraud probability "
                f"({fraud_probability:.4f}) is below "
                f"the decision threshold "
                f"({threshold:.4f})."
            )

# ============================================================
# DATASET SUMMARY
# ============================================================

st.divider()

st.subheader("📈 Dataset Summary")

summary_col1, summary_col2, summary_col3 = st.columns(3)

with summary_col1:

    st.metric(
        "Total Transactions",
        f"{len(df):,}"
    )

with summary_col2:

    st.metric(
        "Normal",
        f"{normal_count:,}"
    )

with summary_col3:

    st.metric(
        "Fraud",
        f"{fraud_count:,}"
    )
