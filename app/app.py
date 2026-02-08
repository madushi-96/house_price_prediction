# app/app.py
# Streamlit House Price Prediction Web App (LKR + no negatives)
# Features:
# - Login page with image
# - Main page: inputs -> predict -> add record -> graphs
# - Clear button to reset inputs
# - Logout returns to login page

# IMPORTANT:
# - Expects a trained sklearn Pipeline saved at: models/house_price_pipeline.joblib
# - Pipeline should include preprocessing (ColumnTransformer) + regressor


# These imports allow us to use modern Python type hints for better code documentation
from __future__ import annotations

# Standard Python utilities for data structures and file handling
from dataclasses import dataclass        # Creates simple classes to store data with less boilerplate code
from pathlib import Path                 # Object-oriented way to work with file system paths (safer than strings)
from typing import Any, Dict, List       # Type hints to specify what kind of data functions expect/return

# External libraries used in this application (must be installed via pip)
import joblib                              # Used to load/save serialized machine learning models
import matplotlib.pyplot as plt            # Plotting library for creating visualizations and graphs
import numpy as np                         # Numerical computing library for mathematical operations
import pandas as pd                        # Data manipulation library for working with tabular data
import streamlit as st                     # Web framework for creating interactive data apps quickly

# ----------------------------- CONFIGURATION -----------------------------
# Configure the Streamlit page settings (must be first Streamlit command)
st.set_page_config(
    page_title="House Price Prediction",  # Title shown in browser tab
    layout="wide"                         # Use full width of browser instead of centered column
)

# Define file paths using pathlib for cross-platform compatibility
PROJECT_ROOT = Path(__file__).resolve().parent.parent  # Get project root: current file -> parent (app/) -> parent (project/)
MODEL_PATH = PROJECT_ROOT / "models" / "house_price_pipeline.joblib"  # Path to saved ML model

# Demo login credentials (hardcoded for demonstration - in production, use secure authentication)
VALID_USERNAME = "admin"
VALID_PASSWORD = "1234"

# Currency conversion rate: converts predicted USD prices to Sri Lankan Rupees
LKR_PER_USD = 300.0  # Exchange rate (should be updated with current rates in production)


# ----------------------------- HELPER FUNCTIONS -----------------------------

@st.cache_resource        # Decorator: caches the loaded model so it's only loaded once (improves performance)
def load_model(model_path: Path):
     # Check if the model file exists before attempting to load
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found at: {model_path}\n"
            f"Make sure you saved the pipeline to models/house_price_pipeline.joblib"
        )
    # Load and return the serialized model using joblib
    return joblib.load(model_path)


def get_expected_columns(pipeline) -> List[str]:
    """
    Return the feature columns the pipeline expects at predict-time.
    Uses ColumnTransformer.feature_names_in_ (available after fitting).
    """
    # Verify the pipeline has the expected structure (named_steps attribute)
    if not hasattr(pipeline, "named_steps") or "preprocessor" not in pipeline.named_steps:
        raise ValueError("Loaded model is not a Pipeline with a 'preprocessor' step.")

    # Extract the preprocessor component from the pipeline
    pre = pipeline.named_steps["preprocessor"]

    # Check if preprocessor has the feature_names_in_ attribute (set during fit)
    if not hasattr(pre, "feature_names_in_"):
        raise ValueError(
            "Preprocessor does not expose feature_names_in_. "
            "Ensure you fitted the ColumnTransformer on a pandas DataFrame."
        )

    # Return the list of expected feature names
    return list(pre.feature_names_in_)


def make_input_df(expected_cols: List[str], values: Dict[str, Any]) -> pd.DataFrame:
    """
    Build a 1-row DataFrame with ALL expected columns.
    Any fields not provided in `values` will be filled with None (treated as missing).
    """
    # Initialize a dictionary with None for all expected columns
    row = {c: None for c in expected_cols}
     # Update the row with actual values provided by the user
    for k, v in values.items():
        if k in row:        # Only set values for columns the model expects
            row[k] = v
     # Convert the dictionary to a single-row DataFrame (required format for sklearn)
    return pd.DataFrame([row])


@dataclass         # Decorator: automatically generates __init__, __repr__, etc.
class AppInputs:
    gr_liv_area: int      # Above ground living area in square feet
    bedrooms: int         # Number of bedrooms above ground
    full_bath: int        # Number of full bathrooms
    overall_qual: int     # Overall material and finish quality (1-10 scale)
    year_built: int       # Original construction year


def init_session_state() -> None:
    # Track if user is logged in (used to show login page vs main page)
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False

    # Store prediction history as a list of dictionaries
    if "records" not in st.session_state:
        st.session_state.records = []  # Each record: dict with inputs + prediction (LKR)

    # Initialize input fields with default values (used when user clicks "Clear")
    if "gr_liv_area" not in st.session_state:
        st.session_state.gr_liv_area = 1500  # Default: 1500 sqft
    if "bedrooms" not in st.session_state:
        st.session_state.bedrooms = 3        # Default: 3 bedrooms
    if "full_bath" not in st.session_state:
        st.session_state.full_bath = 2       # Default: 2 bathrooms
    if "overall_qual" not in st.session_state:
        st.session_state.overall_qual = 5    # Default: medium quality (5/10)
    if "year_built" not in st.session_state:
        st.session_state.year_built = 2000   # Default: built in 2000

    # Store last prediction in LKR
    if "last_prediction_lkr" not in st.session_state:
        st.session_state.last_prediction_lkr = None      # None = no prediction yet


def clear_inputs() -> None:
    st.session_state.gr_liv_area = 1500
    st.session_state.bedrooms = 3
    st.session_state.full_bath = 2
    st.session_state.overall_qual = 5
    st.session_state.year_built = 2000
    st.session_state.last_prediction_lkr = None     # Clear last prediction


def logout() -> None:
    st.session_state.logged_in = False  # Set logged in status to False
    st.session_state.records = []       # Clear prediction history
    clear_inputs()                      # Reset input fields to defaults


# ----------------------------- UI PAGE FUNCTIONS -----------------------------
def login_page() -> None:
     # Display page title and subtitle
    st.title("🏠 House Price Prediction")
    st.caption("Login to continue")

    # Display a house icon image from an external URL
    st.image("https://cdn-icons-png.flaticon.com/512/69/69524.png", width=140)

    # Create text input fields for credentials
    username = st.text_input("Username")  # Regular text input
    password = st.text_input("Password", type="password")  # Password field (hides input)

    # Create two columns for Login and Clear buttons
    col1, col2 = st.columns(2)
    
    with col1:  # Left column: Login button
        if st.button("Login", use_container_width=True):  # Full-width button
            # Validate credentials against hardcoded values
            if username == VALID_USERNAME and password == VALID_PASSWORD:
                st.session_state.logged_in = True  # Set login status
                st.success("✅ Login successful")
                st.rerun()  # Refresh page to show main page
            else:
                st.error("❌ Invalid username or password")
    
    with col2:  # Right column: Clear button
        if st.button("Clear", use_container_width=True):
            st.rerun()  # Refresh page to clear input fields


def main_page(model) -> None:
     # Display page header
    st.title("🏠 House Price Prediction App")
    st.caption("Enter house details → Predict → Add records → View graphs → Logout")

    # Get the list of feature columns the model expects
    expected_cols = get_expected_columns(model)

    # Display informational message about model limitations
    st.info(
        "Note: The model may output negative values if it expects many features but the app provides only a few. "
        "This app converts to LKR and will never show negatives (it clamps to 0)."
    )

    # ---------- INPUT SECTION ----------

    st.subheader("Enter House Details")
    # Create 3 columns for organizing input fields
    c1, c2, c3 = st.columns(3)

    with c1:  # First column: Living area and bedrooms
        # Number input for living area with validation constraints
        st.number_input(
            "Living Area (GrLivArea) - sqft",
            min_value=100,      # Minimum reasonable house size
            max_value=10000,    # Maximum reasonable house size
            step=10,            # Increment by 10
            key="gr_liv_area",  # Links to session_state.gr_liv_area
        )
        # Number input for bedrooms
        st.number_input(
            "Bedrooms (BedroomAbvGr)",
            min_value=0,
            max_value=20,
            step=1,
            key="bedrooms",
        )

    with c2:  # Second column: Bathrooms and quality
        st.number_input(
            "Full Bathrooms (FullBath)",
            min_value=0,
            max_value=10,
            step=1,
            key="full_bath",
        )
        # Quality rating on 1-10 scale
        st.number_input(
            "Overall Quality (OverallQual) (1–10)",
            min_value=1,
            max_value=10,
            step=1,
            key="overall_qual",
        )

    with c3:  # Third column: Year built
        st.number_input(
            "Year Built (YearBuilt)",
            min_value=1800,  # Reasonable historical minimum
            max_value=2026,  # Current year
            step=1,
            key="year_built",
        )

    # Create 3 columns for action buttons
    btn1, btn2, btn3 = st.columns(3)
    with btn1:
        predict_clicked = st.button("Predict Price", use_container_width=True)
    with btn2:
        if st.button("Clear Inputs", use_container_width=True):
            clear_inputs()  # Reset all inputs to defaults
            st.rerun()      # Refresh page to show cleared values
    with btn3:
        if st.button("Logout", use_container_width=True):
            logout()   # Clear session and return to login
            st.rerun()

    # ---------- PREDICTION SECTION ----------
    if predict_clicked:  # Execute when user clicks "Predict Price"
        
        # Create AppInputs object with current values from session state
        inputs = AppInputs(
            gr_liv_area=int(st.session_state.gr_liv_area),
            bedrooms=int(st.session_state.bedrooms),
            full_bath=int(st.session_state.full_bath),
            overall_qual=int(st.session_state.overall_qual),
            year_built=int(st.session_state.year_built),
        )
        
        # Map user inputs to the exact column names the model expects
        values = {
            "GrLivArea": inputs.gr_liv_area,
            "BedroomAbvGr": inputs.bedrooms,
            "FullBath": inputs.full_bath,
            "OverallQual": inputs.overall_qual,
            "YearBuilt": inputs.year_built,
        }
         # Create a DataFrame with all expected columns (missing ones filled with None)
        input_df = make_input_df(expected_cols, values)

        try:
           # pred_usd = float(model.predict(input_df)[0])

            # Convert USD -> LKR
            #pred_lkr = pred_usd * LKR_PER_USD

            import numpy as np
            # Make prediction using the loaded model
            # Model predicts log-transformed price, so we need to reverse the transformation
            pred_log = float(model.predict(input_df)[0])  # Get first (and only) prediction
            
            # Reverse log transformation: expm1(x) = exp(x) - 1 (inverse of log1p)
            pred_usd = float(np.expm1(pred_log))
            
            # Convert USD prediction to Sri Lankan Rupees
            pred_lkr = pred_usd * LKR_PER_USD

            # Handle edge case: ensure price is never negative (not realistic)
            # No negatives allowed
            if pred_lkr < 0:
                st.warning(
                    "Model predicted a negative price (not realistic). "
                    "This happens when many features are missing. Showing 0 instead."
                )
            pred_lkr = max(pred_lkr, 0.0)  # Clamp to minimum of 0

            # Store prediction in session state for later use
            st.session_state.last_prediction_lkr = pred_lkr
            
            # Display prediction result to user with formatting (commas, 2 decimals)
            st.success(f"💰 Predicted Price: LKR {pred_lkr:,.2f}")

        except Exception as e:
            # Catch and display any errors during prediction
            st.error("Prediction failed. See details below.")
            st.exception(e)  # Shows full error traceback

    # ---------- ADD RECORD SECTION ----------
    # Only show "Add to History" button if a prediction has been made
    if st.session_state.last_prediction_lkr is not None:
        if st.button("Add This Prediction to History"):
            # Append current inputs and prediction to records list
            st.session_state.records.append(
                {
                    "GrLivArea": st.session_state.gr_liv_area,
                    "BedroomAbvGr": st.session_state.bedrooms,
                    "FullBath": st.session_state.full_bath,
                    "OverallQual": st.session_state.overall_qual,
                    "YearBuilt": st.session_state.year_built,
                    "PredictedPrice_LKR": st.session_state.last_prediction_lkr,
                }
            )
            st.success("✅ Added to history")

    # ---------- HISTORY + GRAPHS ----------
    st.divider()    # Visual separator
    st.subheader("History & Graphs")
    
    # Only show history section if there are records
    if st.session_state.records:
        # Convert list of dictionaries to DataFrame for easy display and manipulation
        hist_df = pd.DataFrame(st.session_state.records)
        
        # Display the full history as an interactive table
        st.dataframe(hist_df, use_container_width=True)

        # Create 2 columns for side-by-side graphs
        colA, colB = st.columns(2)

        with colA:  # Left column: Line chart of predictions over time
            st.write("📈 Predicted Price (LKR) over Records")
            fig1, ax1 = plt.subplots()  # Create figure and axis objects
            # Plot predicted prices with markers at each point
            ax1.plot(hist_df["PredictedPrice_LKR"], marker="o")
            ax1.set_xlabel("Record #")  # X-axis label
            ax1.set_ylabel("Predicted Price (LKR)")  # Y-axis label
            st.pyplot(fig1)  # Display the plot in Streamlit

        with colB:  # Right column: Scatter plot showing relationship
            st.write("📊 Living Area vs Predicted Price (LKR)")
            fig2, ax2 = plt.subplots()
            # Scatter plot: each point is one prediction
            ax2.scatter(hist_df["GrLivArea"], hist_df["PredictedPrice_LKR"])
            ax2.set_xlabel("GrLivArea (sqft)")
            ax2.set_ylabel("Predicted Price (LKR)")
            st.pyplot(fig2)

        # Create 2 columns for history management buttons
        cbtn1, cbtn2 = st.columns(2)
        
        with cbtn1:
            if st.button("Clear History (Records + Graphs)", use_container_width=True):
                st.session_state.records = []  # Empty the records list
                st.rerun()  # Refresh to update UI
        with cbtn2:
            # Convert DataFrame to CSV format for download
            csv_bytes = hist_df.to_csv(index=False).encode("utf-8")
            # Download button: allows user to save history as CSV file
            st.download_button(
                "Download History as CSV",
                data=csv_bytes,
                file_name="prediction_history.csv",
                mime="text/csv",  # MIME type for CSV files
                use_container_width=True,
            )
    else:
        # Show message when no records exist yet
        st.info("No records yet. Predict and click **Add This Prediction to History** to see graphs.")

# ----------------------------- APPLICATION ENTRY POINT -----------------------------
def main() -> None:
    # Initialize all session state variables (runs on first load)
    init_session_state()

    try:
        # Attempt to load the trained model from disk
        model = load_model(MODEL_PATH)
    except Exception as e:
        # If model loading fails, show error and stop execution
        st.error("❌ Could not load the saved model pipeline.")
        st.exception(e)
        st.stop()  # Halt app execution

    # Route to appropriate page based on login status
    if st.session_state.logged_in:
        main_page(model)  # Show main app if logged in
    else:
        login_page()      # Show login page if not logged in


# Python idiom: only run main() if this file is executed directly (not imported)
if __name__ == "__main__":
    main()