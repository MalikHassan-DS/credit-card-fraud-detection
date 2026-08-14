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
DATASET_PATH = BASE_DIR / "test_transactions.csv"


# ============================================================
# FEATURES
# ============================================================

FEATURES = [
    "Time",
    "V1",
    "V2",
    "V3",
    "V4",
    "V5",
    "V6",
    "V7",
    "V8",
    "V9",
    "V10",
    "V11",
    "V12",
    "V13",
    "V14",
    "V15",
    "V16",
    "V17",
    "V18",
    "V19",
    "V20",
    "V21",
    "V22",
    "V23",
    "V24",
    "V25",
    "V26",
    "V27",
    "V28",
    "Amount"
]


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    model_path = MODEL_DIR / "fraud_detection_random_forest.pkl"
    threshold_path = MODEL_DIR / "threshold.pkl"

    model = joblib.load(model_path)
    threshold = joblib.load(threshold_path)

    return model, threshold


# ============================================================
# LOAD TEST TRANSACTIONS
# ============================================================

@st.cache_data
def load_test_transactions():
    if not DATASET_PATH.exists():
        return None

    return pd.read_csv(DATASET_PATH)


# ============================================================
# LOAD MODEL AND DATASET
# ============================================================

model, threshold = load_model()
df = load_test_transactions()


# ============================================================
# PAGE HEADER
# ============================================================

st.title("💳 Credit Card Fraud Detection System")

st.markdown(
    "This application uses a **Random Forest Machine Learning model** "
    "to detect potentially fraudulent credit card transactions."
)

st.divider()


# ============================================================
# DATASET CHECK
# ============================================================

if df is None:
    st.error(
        "❌ test_transactions.csv was not found. "
        "Please place test_transactions.csv in the same folder as app.py."
    )
    st.stop()


# ============================================================
# DATASET VALIDATION
# ============================================================

required_columns = FEATURES + ["Class"]

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    st.error(
        f"❌ Missing columns in test_transactions.csv: "
        f"{missing_columns}"
    )
    st.stop()


# ============================================================
# MODEL INFORMATION
# ============================================================

col1, col2, col3 = st.columns(3)

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
        "Test Transactions",
        f"{len(df):,}"
    )

st.divider()


# ============================================================
# SESSION STATE
# ============================================================

if "selected_transaction" not in st.session_state:
    st.session_state.selected_transaction = None

if "actual_label" not in st.session_state:
    st.session_state.actual_label = None

if "selected_index" not in st.session_state:
    st.session_state.selected_index = None


# ============================================================
# TEST TRANSACTION LOADER
# ============================================================

st.subheader("📂 Test Dataset Transaction")

st.info(
    "Load a real transaction from test_transactions.csv. "
    "You do not need to manually enter V1–V28."
)

col1, col2 = st.columns(2)


# ============================================================
# LOAD NORMAL TRANSACTION
# ============================================================

with col1:
    if st.button(
        "🟢 Load Normal Transaction",
        use_container_width=True
    ):
        normal_rows = df[df["Class"] == 0]

        if normal_rows.empty:
            st.error(
                "❌ No normal transaction found in test_transactions.csv."
            )
        else:
            selected = normal_rows.sample(
                n=1,
                random_state=None
            ).iloc[0]

            st.session_state.selected_transaction = selected
            st.session_state.actual_label = 0
            st.session_state.selected_index = selected.name

            st.rerun()


# ============================================================
# LOAD FRAUD TRANSACTION
# ============================================================

with col2:
    if st.button(
        "🔴 Load Fraud Transaction",
        use_container_width=True
    ):
        fraud_rows = df[df["Class"] == 1]

        if fraud_rows.empty:
            st.error(
                "❌ No fraud transaction found in test_transactions.csv."
            )
        else:
            selected = fraud_rows.sample(
                n=1,
                random_state=None
            ).iloc[0]

            st.session_state.selected_transaction = selected
            st.session_state.actual_label = 1
            st.session_state.selected_index = selected.name

            st.rerun()


# ============================================================
# TRANSACTION DETAILS
# ============================================================

st.divider()

st.subheader("💳 Transaction Details")

selected_transaction = st.session_state.selected_transaction


# ============================================================
# NO TRANSACTION SELECTED
# ============================================================

if selected_transaction is None:
    st.info(
        "Choose **Load Normal Transaction** or "
        "**Load Fraud Transaction** to automatically "
        "fill the transaction values."
    )
    st.stop()


# ============================================================
# ACTUAL LABEL
# ============================================================

actual_label = int(st.session_state.actual_label)

if actual_label == 0:
    st.success("Actual Label: 🟢 NORMAL (0)")
else:
    st.warning("Actual Label: 🔴 FRAUD (1)")


# ============================================================
# INPUT DATA
# ============================================================

input_data = {}


# ============================================================
# TRANSACTION INFORMATION
# ============================================================

st.subheader("Transaction Information")

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


# ============================================================
# V1 - V14
# ============================================================

st.subheader("Transaction Features V1 - V14")

cols = st.columns(3)

for i, feature in enumerate(FEATURES[1:15]):
    with cols[i % 3]:
        input_data[feature] = st.number_input(
            feature,
            value=float(selected_transaction[feature]),
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
            value=float(selected_transaction[feature]),
            format="%.6f",
            key=f"{feature}_input"
        )


# ============================================================
# VIEW COMPLETE TRANSACTION
# ============================================================

with st.expander("📄 View Complete Transaction Data"):
    transaction_view = pd.DataFrame(
        [selected_transaction[FEATURES].to_dict()]
    )

    st.dataframe(
        transaction_view,
        use_container_width=True
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
    # GET MODEL PROBABILITIES
    # --------------------------------------------------------

    probabilities = model.predict_proba(transaction)[0]

    classes = list(model.classes_)


    # --------------------------------------------------------
    # FRAUD PROBABILITY
    # --------------------------------------------------------

    if 1 in classes:
        fraud_probability = float(
            probabilities[classes.index(1)]
        )
    else:
        fraud_probability = 0.0


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


    st.divider()


    # ========================================================
    # ACTUAL VS PREDICTED
    # ========================================================

    st.subheader("📊 Actual vs Predicted")

    col1, col2 = st.columns(2)


    # --------------------------------------------------------
    # ACTUAL LABEL
    # --------------------------------------------------------

    with col1:
        if actual_label == 1:
            st.error(
                "Actual Label: 🔴 FRAUD (1)"
            )
        else:
            st.success(
                "Actual Label: 🟢 NORMAL (0)"
            )


    # --------------------------------------------------------
    # PREDICTED LABEL
    # --------------------------------------------------------

    with col2:
        if prediction == 1:
            st.error(
                "Predicted Label: 🔴 FRAUD (1)"
            )
        else:
            st.success(
                "Predicted Label: 🟢 NORMAL (0)"
            )


    # ========================================================
    # CORRECT / INCORRECT
    # ========================================================

    if actual_label == prediction:
        st.success(
            "✅ Model Prediction is CORRECT"
        )
    else:
        st.warning(
            "⚠️ Model Prediction is INCORRECT"
        )
