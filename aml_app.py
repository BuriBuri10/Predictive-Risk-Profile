# app.py - Final Version with Sidebar Layout

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import xgboost
import json
from typing import List

# --- 1. LANGCHAIN AND GEMINI IMPORTS ---
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field

# --- HELPER FUNCTION ---
def find_column_by_keyword(df, keyword):
    """Finds the first column name containing a keyword, case-insensitive."""
    for col in df.columns:
        if keyword.lower() in col.lower():
            return col
    return None

# --- 2. INITIAL SETUP AND DATA LOADING ---
st.set_page_config(layout="wide")

@st.cache_data
def load_model_and_columns():
    model = joblib.load('aml_xgb_model.pkl')
    model_columns = joblib.load('model_columns.pkl')
    return model, model_columns

@st.cache_data
def load_full_dataset():
    try:
        df = pd.read_csv("C://0//Evrything//TrainLikeHell//SAML-D.csv//SAML-D.csv") # UPDATE THIS PATH
        date_col = find_column_by_keyword(df, 'date')
        time_col = find_column_by_keyword(df, 'time')
        if date_col and time_col:
            df['Timestamp'] = pd.to_datetime(df[date_col].astype(str) + ' ' + df[time_col].astype(str))
            df = df.drop(columns=[date_col, time_col])
        return df
    except FileNotFoundError:
        st.error("FATAL: The dataset file was not found. Please check the file path.")
        return pd.DataFrame()

# Load data and model
model, model_columns = load_model_and_columns()
full_df = load_full_dataset()

# (LangChain/Gemini implementation remains the same)
class CustomerRiskProfile(BaseModel):
    overall_risk_level: str = Field(description="The final aggregated risk level for the customer. Must be one of: 'Low', 'Medium', 'High'.")
    primary_risk_factors: List[str] = Field(description="A Python list of key phrases identifying the top 2-3 risk factors.")
    narrative_summary: str = Field(description="A concise, one-paragraph narrative summary of the customer's profile and behavior.")
    recommended_action: str = Field(description="The suggested next step. Must be one of: 'Standard Monitoring', 'Requires Manual Review', 'Flag for Investigation'.")

def get_llm_profile_summary(account_id, profile_stats):
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", google_api_key=st.secrets["GEMINI_API_KEY"], temperature=0.2)
        parser = JsonOutputParser(pydantic_object=CustomerRiskProfile)
        prompt_template = """You are a senior financial risk analyst... (rest of the prompt is the same)
        {format_instructions}"""
        prompt = ChatPromptTemplate.from_template(template=prompt_template, partial_variables={"format_instructions": parser.get_format_instructions()})
        chain = prompt | llm | parser
        llm_result = chain.invoke({"account_id": account_id, "profile_stats": json.dumps(profile_stats, indent=2)})
        return llm_result
    except Exception as e:
        st.error(f"An error occurred with the AI model: {e}")
        return None

# --- 4. UI: MAIN AREA AND SIDEBAR ---
st.title("👤 Customer Risk Profiling Dashboard")
st.sidebar.title("Search & Analysis")
st.sidebar.write("Enter an existing customer account ID to generate a comprehensive risk profile.")

account_id_input = st.sidebar.text_input(
    "Enter Customer Account ID",
    placeholder="e.g.- 9139420948"
)

if st.sidebar.button("Generate Risk Profile"):
    # --- 5. INPUT VALIDATION AND PROFILE GENERATION ---
    main_area = st.container() # Create a container for results

    # 1. Validate input format
    if not account_id_input.isdigit() or not (5 <= len(account_id_input) <= 12):
        main_area.error("Invalid format. Please enter a numeric Account ID between 5 and 12 digits.")
    else:
        account_to_profile = int(account_id_input)
        
        # Programmatically find account columns
        sender_account_col = find_column_by_keyword(full_df, 'sender_account')
        receiver_account_col = find_column_by_keyword(full_df, 'receiver_account')
        
        # 2. Check if account exists
        is_sender = full_df[sender_account_col] == account_to_profile
        is_receiver = full_df[receiver_account_col] == account_to_profile
        
        if not is_sender.any() and not is_receiver.any():
            main_area.warning(f"Account ID '{account_to_profile}' not found in the transaction dataset.")
        else:
            # 3. If valid and exists, generate the profile
            with st.spinner(f"Generating profile for Account {account_to_profile}..."):
                customer_df = full_df[is_sender | is_receiver].copy()
                
                # Find all necessary columns for feature engineering within this scope
                timestamp_col = find_column_by_keyword(customer_df, 'timestamp')
                amount_col = find_column_by_keyword(customer_df, 'amount')
                payment_currency_col = find_column_by_keyword(customer_df, 'payment_currency')
                received_currency_col = find_column_by_keyword(customer_df, 'received_currency')
                sender_bank_loc_col = find_column_by_keyword(customer_df, 'sender_bank_location')
                receiver_bank_loc_col = find_column_by_keyword(customer_df, 'receiver_bank_location')
                is_laundering_col = find_column_by_keyword(customer_df, 'is_laundering')
                laundering_type_col = find_column_by_keyword(customer_df, 'laundering_type')

                # Feature Engineering
                customer_df['hour_of_day'] = customer_df[timestamp_col].dt.hour
                customer_df['day_of_week'] = customer_df[timestamp_col].dt.dayofweek
                customer_df['is_foreign_exchange'] = (customer_df[payment_currency_col] != customer_df[received_currency_col]).astype(int)
                customer_df['is_cross_border'] = (customer_df[sender_bank_loc_col] != customer_df[receiver_bank_loc_col]).astype(int)
                customer_df['log_amount'] = np.log1p(customer_df[amount_col])
                customer_df['sender_transaction_count'] = len(customer_df[customer_df[sender_account_col] == account_to_profile])
                customer_df['sender_total_amount'] = customer_df[customer_df[sender_account_col] == account_to_profile][amount_col].sum()
                customer_df['receiver_transaction_count'] = len(customer_df[customer_df[receiver_account_col] == account_to_profile])
                customer_df['receiver_total_amount'] = customer_df[customer_df[receiver_account_col] == account_to_profile][amount_col].sum()
                customer_df['sender_fan_out'] = customer_df[customer_df[sender_account_col] == account_to_profile][receiver_account_col].nunique()
                customer_df['receiver_fan_in'] = customer_df[customer_df[receiver_account_col] == account_to_profile][sender_account_col].nunique()

                # Prepare for prediction
                X_customer = customer_df.drop(columns=[is_laundering_col, laundering_type_col, timestamp_col, sender_account_col, receiver_account_col])
                X_customer_encoded = pd.get_dummies(X_customer, columns=[payment_currency_col, received_currency_col, sender_bank_loc_col, receiver_bank_loc_col], drop_first=True)
                X_customer_aligned = X_customer_encoded.reindex(columns=model_columns, fill_value=0)
                
                # Predictions and Aggregations
                predictions = model.predict(X_customer_aligned)
                probabilities = model.predict_proba(X_customer_aligned)[:, 1]
                customer_df['risk_score'] = probabilities
                
                high_risk_txns = customer_df[customer_df['risk_score'] >= 0.5]
                
                profile_stats = {
                    "total_transactions": len(customer_df), "high_risk_transaction_count": len(high_risk_txns),
                    "percentage_high_risk": f"{len(high_risk_txns) / len(customer_df) if len(customer_df) > 0 else 0:.1%}",
                    "max_risk_score": f"{customer_df['risk_score'].max():.1%}" if len(customer_df) > 0 else "N/A",
                    "average_risk_score": f"{customer_df['risk_score'].mean():.1%}" if len(customer_df) > 0 else "N/A",
                    "total_sent": customer_df['sender_total_amount'].iloc[0] if not customer_df.empty else 0,
                    "total_received": customer_df['receiver_total_amount'].iloc[0] if not customer_df.empty else 0,
                    "unique_counterparties": pd.concat([customer_df[sender_account_col], customer_df[receiver_account_col]]).nunique() -1
                }
                
                # Get Gemini analysis and display dashboard
                llm_summary = get_llm_profile_summary(account_to_profile, profile_stats)
                if llm_summary:
                    main_area.subheader(f"AI-Generated Risk Profile for Account: {account_to_profile}")
                    col1, col2 = main_area.columns(2)
                    with col1:
                        if llm_summary['overall_risk_level'] == 'High': st.error(f"**Overall Risk:** {llm_summary['overall_risk_level']}")
                        elif llm_summary['overall_risk_level'] == 'Medium': st.warning(f"**Overall Risk:** {llm_summary['overall_risk_level']}")
                        else: st.success(f"**Overall Risk:** {llm_summary['overall_risk_level']}")
                    with col2: st.info(f"**Recommended Action:** {llm_summary['recommended_action']}")
                    main_area.markdown("**Narrative Summary:**"); main_area.write(llm_summary['narrative_summary'])
                    main_area.markdown("**Primary Risk Factors Identified:**")
                    for factor in llm_summary['primary_risk_factors']: main_area.markdown(f"- {factor}")
                    main_area.subheader("Profile Statistics")
                    m1, m2, m3, m4 = main_area.columns(4)
                    m1.metric("Total Transactions", profile_stats['total_transactions']); m2.metric("High-Risk Transactions", profile_stats['high_risk_transaction_count'])
                    m3.metric("Max Single Transaction Risk", profile_stats['max_risk_score']); m4.metric("Avg. Transaction Risk", profile_stats['average_risk_score'])
                    main_area.subheader("Top 5 Riskiest Transactions"); main_area.dataframe(customer_df[[timestamp_col, amount_col, 'risk_score']].sort_values(by='risk_score', ascending=False).head(5))

else:
    st.info("Please enter an Account ID in the sidebar to generate a risk profile.")


# # app.py - Final Version (Without Payment Type)

# import streamlit as st
# import pandas as pd
# import numpy as np
# import joblib
# import xgboost
# import json
# from typing import List

# # --- 1. LANGCHAIN AND GEMINI IMPORTS ---
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import JsonOutputParser
# from langchain_core.pydantic_v1 import BaseModel, Field

# # --- HELPER FUNCTION ---
# def find_column_by_keyword(df, keyword):
#     """Finds the first column name containing a keyword, case-insensitive."""
#     for col in df.columns:
#         if keyword.lower() in col.lower():
#             return col
#     return None

# # --- 2. INITIAL SETUP AND DATA LOADING ---
# st.set_page_config(layout="wide")

# @st.cache_data
# def load_model_and_columns():
#     model = joblib.load('aml_xgb_model.pkl')
#     model_columns = joblib.load('model_columns.pkl')
#     return model, model_columns

# @st.cache_data
# def load_full_dataset():
#     try:
#         df = pd.read_csv("C://0//Evrything//TrainLikeHell//SAML-D.csv//SAML-D.csv")
#         date_col = find_column_by_keyword(df, 'date')
#         time_col = find_column_by_keyword(df, 'time')
#         if date_col and time_col:
#             df['Timestamp'] = pd.to_datetime(df[date_col].astype(str) + ' ' + df[time_col].astype(str))
#             df = df.drop(columns=[date_col, time_col])
#         return df
#     except FileNotFoundError:
#         st.error("FATAL: The dataset file was not found. Please check the file path.")
#         return pd.DataFrame()

# # Load data and model
# model, model_columns = load_model_and_columns()
# full_df = load_full_dataset()

# # (LangChain and Gemini implementation remains the same)
# class CustomerRiskProfile(BaseModel):
#     overall_risk_level: str = Field(description="The final aggregated risk level for the customer. Must be one of: 'Low', 'Medium', 'High'.")
#     primary_risk_factors: List[str] = Field(description="A Python list of key phrases identifying the top 2-3 risk factors.")
#     narrative_summary: str = Field(description="A concise, one-paragraph narrative summary of the customer's profile and behavior.")
#     recommended_action: str = Field(description="The suggested next step. Must be one of: 'Standard Monitoring', 'Requires Manual Review', 'Flag for Investigation'.")

# def get_llm_profile_summary(account_id, profile_stats):
#     try:
#         llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", google_api_key=st.secrets["GEMINI_API_KEY"], temperature=0.2)
#         parser = JsonOutputParser(pydantic_object=CustomerRiskProfile)
#         prompt_template = """You are a senior financial risk analyst... (rest of the prompt is the same)
#         {format_instructions}"""
#         prompt = ChatPromptTemplate.from_template(template=prompt_template, partial_variables={"format_instructions": parser.get_format_instructions()})
#         chain = prompt | llm | parser
#         llm_result = chain.invoke({"account_id": account_id, "profile_stats": json.dumps(profile_stats, indent=2)})
#         return llm_result
#     except Exception as e:
#         st.error(f"An error occurred with the AI model: {e}")
#         return None

# # --- 4. DYNAMIC SEARCH UI ---
# st.title("👤 Customer Risk Profiling Dashboard")
# st.write("Use the advanced search panel to find a customer based on their account ID or transaction patterns.")
# st.sidebar.header("Advanced Customer Search")

# # Programmatically find column names for UI widgets
# amount_col = find_column_by_keyword(full_df, 'amount')
# sender_account_col = find_column_by_keyword(full_df, 'sender_account')
# receiver_account_col = find_column_by_keyword(full_df, 'receiver_account')
# timestamp_col = find_column_by_keyword(full_df, 'timestamp')

# with st.sidebar.expander("Search by Criteria", expanded=True):
#     account_id_search = st.text_input("Search by Account ID (exact match)")
#     if timestamp_col:
#         start_date = pd.to_datetime(st.date_input("Transaction Start Date", value=full_df[timestamp_col].min().date()))
#         end_date = pd.to_datetime(st.date_input("Transaction End Date", value=full_df[timestamp_col].max().date()))
#     if amount_col:
#         min_amount = st.number_input("Minimum Transaction Amount", min_value=0.0)
#         max_amount = st.number_input("Maximum Transaction Amount", min_value=0.0, value=float(full_df[amount_col].max()))
    
#     # NOTE: The Payment Type filter has been removed.
    
# search_button = st.sidebar.button("Search for Customers")

# # --- 5. SEARCH AND PROFILE GENERATION LOGIC ---
# if 'search_results' not in st.session_state: st.session_state.search_results = pd.DataFrame()
# if 'selected_account' not in st.session_state: st.session_state.selected_account = None

# if search_button:
#     with st.spinner("Searching for matching customers..."):
#         results_df = full_df.copy()
#         if account_id_search:
#             try:
#                 search_id = int(account_id_search)
#                 results_df = results_df[(results_df[sender_account_col] == search_id) | (results_df[receiver_account_col] == search_id)]
#             except (ValueError, TypeError):
#                 st.sidebar.error("Please enter a valid numeric Account ID.")
#                 results_df = pd.DataFrame()
        
#         # NOTE: The filtering logic for Payment Type has been removed.

#         if amount_col:
#             results_df = results_df[results_df[amount_col].between(min_amount, max_amount)]
#         if timestamp_col:
#             results_df = results_df[results_df[timestamp_col].between(start_date, end_date)]
        
#         st.session_state.search_results = results_df
#         st.session_state.selected_account = None

# if not st.session_state.search_results.empty:
#     result_accounts = pd.concat([st.session_state.search_results[sender_account_col], st.session_state.search_results[receiver_account_col]]).unique()
#     st.session_state.selected_account = st.selectbox(f"Found {len(result_accounts)} matching account(s). Select one:", options=result_accounts, index=None)

# if st.session_state.selected_account:
#     if st.button(f"Generate Profile for {st.session_state.selected_account}"):
#         account_to_profile = st.session_state.selected_account
#         with st.spinner(f"Generating profile for Account {account_to_profile}..."):
#             # This is the full, final logic for generating and displaying the profile
#             customer_df = full_df[(full_df[sender_account_col] == account_to_profile) | (full_df[receiver_account_col] == account_to_profile)].copy()
            
#             if customer_df.empty:
#                 st.warning("No transaction data found for this account.")
#             else:
#                 # Find all necessary columns for feature engineering
#                 payment_currency_col = find_column_by_keyword(customer_df, 'payment_currency')
#                 received_currency_col = find_column_by_keyword(customer_df, 'received_currency')
#                 sender_bank_loc_col = find_column_by_keyword(customer_df, 'sender_bank_location')
#                 receiver_bank_loc_col = find_column_by_keyword(customer_df, 'receiver_bank_location')
#                 is_laundering_col = find_column_by_keyword(customer_df, 'is_laundering')
#                 laundering_type_col = find_column_by_keyword(customer_df, 'laundering_type')

#                 # Feature Engineering
#                 customer_df['hour_of_day'] = customer_df[timestamp_col].dt.hour
#                 customer_df['day_of_week'] = customer_df[timestamp_col].dt.dayofweek
#                 customer_df['is_foreign_exchange'] = (customer_df[payment_currency_col] != customer_df[received_currency_col]).astype(int)
#                 customer_df['is_cross_border'] = (customer_df[sender_bank_loc_col] != customer_df[receiver_bank_loc_col]).astype(int)
#                 customer_df['log_amount'] = np.log1p(customer_df[amount_col])
#                 customer_df['sender_transaction_count'] = len(customer_df[customer_df[sender_account_col] == account_to_profile])
#                 customer_df['sender_total_amount'] = customer_df[customer_df[sender_account_col] == account_to_profile][amount_col].sum()
#                 customer_df['receiver_transaction_count'] = len(customer_df[customer_df[receiver_account_col] == account_to_profile])
#                 customer_df['receiver_total_amount'] = customer_df[customer_df[receiver_account_col] == account_to_profile][amount_col].sum()
#                 customer_df['sender_fan_out'] = customer_df[customer_df[sender_account_col] == account_to_profile][receiver_account_col].nunique()
#                 customer_df['receiver_fan_in'] = customer_df[customer_df[receiver_account_col] == account_to_profile][sender_account_col].nunique()

#                 # Prepare for prediction
#                 X_customer = customer_df.drop(columns=[is_laundering_col, laundering_type_col, timestamp_col, sender_account_col, receiver_account_col])
#                 X_customer_encoded = pd.get_dummies(X_customer, columns=[payment_currency_col, received_currency_col, sender_bank_loc_col, receiver_bank_loc_col], drop_first=True)
#                 X_customer_aligned = X_customer_encoded.reindex(columns=model_columns, fill_value=0)
                
#                 # Predictions and Aggregations
#                 predictions = model.predict(X_customer_aligned)
#                 probabilities = model.predict_proba(X_customer_aligned)[:, 1]
#                 customer_df['risk_score'] = probabilities
#                 customer_df['is_high_risk'] = predictions
#                 high_risk_txns = customer_df[customer_df['is_high_risk'] == 1]
                
#                 profile_stats = {
#                     "total_transactions": len(customer_df), "high_risk_transaction_count": len(high_risk_txns),
#                     "percentage_high_risk": f"{len(high_risk_txns) / len(customer_df) if len(customer_df) > 0 else 0:.1%}",
#                     "max_risk_score": f"{customer_df['risk_score'].max():.1%}" if len(customer_df) > 0 else "N/A",
#                     "average_risk_score": f"{customer_df['risk_score'].mean():.1%}" if len(customer_df) > 0 else "N/A",
#                     "total_sent": profile_stats.get('sender_total_amount', 0),
#                     "total_received": profile_stats.get('receiver_total_amount', 0),
#                     "unique_counterparties": pd.concat([customer_df[sender_account_col], customer_df[receiver_account_col]]).nunique() -1
#                 }
                
#                 # Get Gemini analysis and display dashboard
#                 llm_summary = get_llm_profile_summary(account_to_profile, profile_stats)
#                 if llm_summary:
#                     st.subheader(f"AI-Generated Risk Profile for Account: {account_to_profile}")
#                     # ... (The full dashboard display logic from the previous script) ...
#                     col1, col2 = st.columns(2)
#                     with col1:
#                         if llm_summary['overall_risk_level'] == 'High': st.error(f"**Overall Risk:** {llm_summary['overall_risk_level']}")
#                         elif llm_summary['overall_risk_level'] == 'Medium': st.warning(f"**Overall Risk:** {llm_summary['overall_risk_level']}")
#                         else: st.success(f"**Overall Risk:** {llm_summary['overall_risk_level']}")
#                     with col2: st.info(f"**Recommended Action:** {llm_summary['recommended_action']}")
#                     st.markdown("**Narrative Summary:**"); st.write(llm_summary['narrative_summary'])
#                     st.markdown("**Primary Risk Factors Identified:**")
#                     for factor in llm_summary['primary_risk_factors']: st.markdown(f"- {factor}")
#                     st.subheader("Profile Statistics")
#                     m1, m2, m3, m4 = st.columns(4)
#                     m1.metric("Total Transactions", profile_stats['total_transactions']); m2.metric("High-Risk Transactions", profile_stats['high_risk_transaction_count'])
#                     m3.metric("Max Single Transaction Risk", profile_stats['max_risk_score']); m4.metric("Avg. Transaction Risk", profile_stats['average_risk_score'])
#                     st.subheader("Top 5 Riskiest Transactions"); st.dataframe(customer_df[[timestamp_col, amount_col, 'risk_score']].sort_values(by='risk_score', ascending=False).head(5))

# else:
#     st.info("Use the sidebar to search for a customer to begin building a profile.")



# # app.py
# import streamlit as st
# import pandas as pd
# import numpy as np
# import joblib
# import xgboost
# import json
# from typing import List
# from pydantic import BaseModel, Field

# # --- 1. INITIAL SETUP AND DATA LOADING ---
# st.set_page_config(layout="wide")

# @st.cache_data
# def load_model_and_columns():
#     # Your model loading code here...
#     model = joblib.load('aml_xgb_model.pkl')
#     model_columns = joblib.load('model_columns.pkl')
#     return model, model_columns

# model, model_columns = load_model_and_columns()

# @st.cache_data
# def load_full_dataset():
#     # Your data loading code here...
#     df = pd.read_csv("C://0//Evrything//TrainLikeHell//SAML-D.csv//SAML-D.csv")
#     df['Timestamp'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['Time'].astype(str))
#     df = df.drop(columns=['Date', 'Time'])
#     return df

# full_df = load_full_dataset()

# # (The LangChain/Gemini implementation remains the same)
# class CustomerRiskProfile(BaseModel):
#     overall_risk_level: str = Field(description="The final aggregated risk level for the customer. Must be one of: 'Low', 'Medium', 'High'.")
#     primary_risk_factors: List[str] = Field(description="A Python list of key phrases identifying the top 2-3 risk factors.")
#     narrative_summary: str = Field(description="A concise, one-paragraph narrative summary of the customer's profile and behavior.")
#     recommended_action: str = Field(description="The suggested next step. Must be one of: 'Standard Monitoring', 'Requires Manual Review', 'Flag for Investigation'.")

# def get_llm_profile_summary(account_id, profile_stats):
#     # Your get_llm_profile_summary function here...
#     try:
#         llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", google_api_key=st.secrets["GEMINI_API_KEY"], temperature=0.2)
#         parser = JsonOutputParser(pydantic_object=CustomerRiskProfile)
#         prompt_template = """
#         You are a senior financial risk analyst... (rest of the prompt is the same)
#         {format_instructions}
#         """
#         prompt = ChatPromptTemplate.from_template(template=prompt_template, partial_variables={"format_instructions": parser.get_format_instructions()})
#         chain = prompt | llm | parser
#         llm_result = chain.invoke({"account_id": account_id, "profile_stats": json.dumps(profile_stats, indent=2)})
#         return llm_result
#     except Exception as e:
#         st.error(f"An error occurred with the AI model: {e}")
#         return None

# # --- 4. NEW DYNAMIC SEARCH UI ---
# st.title("👤 Customer Risk Profiling Dashboard")
# st.write("Use the advanced search panel to find a customer and generate a comprehensive risk profile.")

# st.sidebar.header("Advanced Customer Search")

# # We use an expander for a clean UI
# with st.sidebar.expander("Search by Criteria", expanded=True):
#     # Assuming 'Name' is a column in your dataset. If not, this can be adapted.
#     # For this example, we'll create a dummy name based on account ID.
#     # In a real scenario, you'd have a 'Customer_Name' column.
#     customer_name_search = st.text_input("Customer Name (contains)")
    
#     account_id_search = st.text_input("Account ID (exact match)")

#     min_amount = st.number_input("Minimum Transaction Amount", min_value=0.0)
#     max_amount = st.number_input("Maximum Transaction Amount", min_value=0.0, value=100000.0)

#     payment_type_filter = st.multiselect("Filter by Payment Type", options=full_df['Payment_type'].unique())
    
#     start_date = st.date_input("Transaction Start Date", value=pd.to_datetime('2022-01-01'))
#     end_date = st.date_input("Transaction End Date", value=pd.to_datetime('2025-12-31'))

# search_button = st.sidebar.button("Search for Customers")


# # --- 5. SEARCH AND PROFILE GENERATION LOGIC ---

# # Initialize a session state to hold search results
# if 'search_results' not in st.session_state:
#     st.session_state.search_results = pd.DataFrame()

# if search_button:
#     with st.spinner("Searching for matching customers..."):
#         # Start with the full dataset
#         results_df = full_df.copy()

#         # In a real dataset, you'd have a name column. We simulate it here.
#         # This is just for demonstration.
#         if customer_name_search:
#              # This is a placeholder for a real name search.
#              # We'll pretend names are like "Customer_12345"
#              results_df['Name'] = "Customer_" + results_df['Sender_account'].astype(str)
#              results_df = results_df[results_df['Name'].str.contains(customer_name_search, case=False)]
        
#         # Filter based on other criteria
#         if account_id_search:
#             results_df = results_df[
#                 (results_df['Sender_account'] == int(account_id_search)) | 
#                 (results_df['Receiver_account'] == int(account_id_search))
#             ]

#         if payment_type_filter:
#             results_df = results_df[results_df['Payment_type'].isin(payment_type_filter)]
        
#         # Filter by amount and date
#         results_df = results_df[results_df['Amount'].between(min_amount, max_amount)]
#         results_df = results_df[results_df['Timestamp'].between(pd.to_datetime(start_date), pd.to_datetime(end_date))]
        
#         st.session_state.search_results = results_df

# # Display search results and allow user to select an account
# if not st.session_state.search_results.empty:
#     st.subheader("Search Results")
#     st.write(f"Found {len(st.session_state.search_results)} matching transactions for the following accounts.")

#     # Get unique accounts from the search results
#     result_accounts = pd.concat([
#         st.session_state.search_results['Sender_account'],
#         st.session_state.search_results['Receiver_account']
#     ]).unique()
    
#     selected_account_from_search = st.selectbox("Select an account from the results to generate a profile:", options=result_accounts)

#     if st.button(f"Generate Profile for {selected_account_from_search}"):
#         # This section is the same as your previous "Generate Risk Profile" button logic
#         with st.spinner(f"Generating profile for Account {selected_account_from_search}..."):
#             # (The entire profile generation logic from Step 5 of the previous script goes here)
#             # a) Filter dataset...
#             customer_df = full_df[(full_df['Sender_account'] == selected_account_from_search) | (full_df['Receiver_account'] == selected_account_from_search)].copy()
#             # b) Apply feature engineering...
#             # c) One-hot encode...
#             # d) Get predictions...
#             # e) Aggregate statistics...
#             # f) Get the final analysis from Gemini...
#             # g) Display dashboard...
#             st.success("Profile generated successfully! (Implementation logic for display goes here)")
#             # You would paste your detailed display logic here, using st.metric, st.info, etc.


# else:
#     st.info("Use the sidebar to search for a customer to begin.")