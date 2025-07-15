import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime

from utils.cleaner import clean_transaction_data
from models.forecast import forecast_spending, cluster_transactions
from database import (
    initialize_database, insert_transactions, get_summary_by_category,
    get_monthly_summary, detect_unusual_transactions, get_spending_trends
)

# Initialize session state for database
if 'db_initialized' not in st.session_state:
    st.session_state.db_initialized = False

# Initialize database only once per session
if not st.session_state.db_initialized:
    initialize_database()
    st.session_state.db_initialized = True

# Page configuration
st.set_page_config(
    page_title="Smart Budgeting Tracker",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main title
st.title("Smart Budgeting Tracker")

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Choose a page", [
    "Dashboard", 
    "Upload Data", 
    "Spending Analysis", 
    "Forecasting", 
    "Alerts & Insights"
])

# Data status indicator
st.sidebar.markdown("---")
st.sidebar.subheader("Data Status")

try:
    summary = get_summary_by_category()
    if not summary.empty:
        total_transactions = summary['transaction_count'].sum()
        st.sidebar.success(f"{total_transactions} transactions loaded")
    else:
        st.sidebar.warning("No transaction data")
except Exception as e:
    st.sidebar.error("Database error")

if page == "Upload Data":
    st.header("Upload Your Financial Data")
    
    # File upload section
    uploaded_file = st.file_uploader(
        "Upload your bank statement",
        type=['csv', 'xlsx', 'xls'],
        help="Supported formats: CSV, Excel (.xlsx, .xls)"
    )
    
    if uploaded_file:
        try:
            # Read the file based on its type
            if uploaded_file.name.endswith('.csv'):
                raw_df = pd.read_csv(uploaded_file)
            else:
                raw_df = pd.read_excel(uploaded_file)
            
            st.subheader("Raw Data Preview")
            st.dataframe(raw_df.head())
            
            # Clean the data
            with st.spinner("Cleaning and processing data..."):
                cleaned_df = clean_transaction_data(raw_df)
                
            st.subheader("Cleaned Data")
            st.dataframe(cleaned_df)
            
            st.info(f"Ready to insert {len(cleaned_df)} transactions")
            
            # Insert into database
            if st.button("Save to Database", type="primary"):
                with st.spinner("Saving transactions..."):
                    inserted_count = insert_transactions(cleaned_df)
                    if inserted_count > 0:
                        st.success(f"Successfully saved {inserted_count} new transactions!")
                    else:
                        st.info("No new transactions to save (duplicates filtered out)")
                
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            st.info("Please check that your file has columns for date, description, and amount")
    
    # Manual transaction entry
    st.subheader("Add Manual Transaction")
    with st.form("manual_transaction"):
        col1, col2 = st.columns(2)
        
        with col1:
            date = st.date_input("Date", datetime.now())
            amount = st.number_input("Amount", value=0.0, help="Use negative for expenses, positive for income")
        
        with col2:
            description = st.text_input("Description")
            category = st.selectbox("Category", [
                'groceries', 'rent', 'utilities', 'entertainment', 'food',
                'transportation', 'income', 'healthcare', 'shopping', 'education',
                'savings', 'debt', 'other'
            ])
        
        notes = st.text_area("Notes (optional)")
        
        if st.form_submit_button("Add Transaction"):
            if description and amount != 0:
                manual_df = pd.DataFrame({
                    'date': [date],
                    'description': [description],
                    'amount': [amount],
                    'category': [category],
                    'subcategory': [None],
                    'notes': [notes],
                    'is_recurring': [False]
                })
                
                insert_transactions(manual_df)
                st.success("Transaction added successfully!")
                st.rerun()
            else:
                st.error("Please fill in description and amount")

elif page == "Dashboard":
    st.header("Financial Dashboard")
    
    # Get summary data
    category_summary = get_summary_by_category()
    monthly_summary = get_monthly_summary()
    
    if not category_summary.empty:
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        total_income = category_summary[category_summary['total_amount'] > 0]['total_amount'].sum()
        total_expenses = abs(category_summary[category_summary['total_amount'] < 0]['total_amount'].sum())
        net_worth = total_income - total_expenses
        
        if not monthly_summary.empty:
            avg_monthly_spending = total_expenses / max(1, len(monthly_summary['month'].unique()))
        else:
            avg_monthly_spending = 0
        
        with col1:
            st.metric("Total Income", f"${total_income:,.2f}")
        with col2:
            st.metric("Total Expenses", f"${total_expenses:,.2f}")
        with col3:
            st.metric("Net Worth", f"${net_worth:,.2f}")
        with col4:
            st.metric("Avg Monthly Spending", f"${avg_monthly_spending:,.2f}")
        
        # Spending breakdown
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Spending by Category")
            expenses = category_summary[category_summary['total_amount'] < 0].copy()
            expenses['total_amount'] = expenses['total_amount'].abs()
            
            if not expenses.empty:
                fig = px.pie(
                    expenses,
                    values='total_amount',
                    names='category',
                    title="Expense Distribution"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No expense data available")
        
        with col2:
            st.subheader("Income vs Expenses")
            if total_income > 0 or total_expenses > 0:
                income_expense_data = {
                    'Type': ['Income', 'Expenses'],
                    'Amount': [total_income, total_expenses]
                }
                
                fig = px.bar(
                    income_expense_data,
                    x='Type',
                    y='Amount',
                    color='Type',
                    title="Income vs Expenses"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No income/expense data available")
        
        # Monthly trends
        if not monthly_summary.empty:
            st.subheader("Monthly Spending Trends")
            
            # Filter for expenses only
            expense_trends = monthly_summary[monthly_summary['total_amount'] < 0].copy()
            expense_trends['total_amount'] = expense_trends['total_amount'].abs()
            
            if not expense_trends.empty:
                fig = px.line(
                    expense_trends,
                    x='month',
                    y='total_amount',
                    color='category',
                    title="Monthly Spending by Category"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No monthly expense trends available")
    
    else:
        st.info("No transaction data available. Please upload your bank statement in the 'Upload Data' section.")

elif page == "Spending Analysis":
    st.header("Spending Analysis")
    
    category_summary = get_summary_by_category()
    
    if not category_summary.empty:
        # Transaction clustering
        st.subheader("Transaction Clustering")
        
        try:
            clusters = cluster_transactions()
            if not clusters.empty:
                st.write("K-means clustering has grouped your transactions into similar patterns:")
                
                # Show cluster summary
                cluster_summary = clusters.groupby('cluster').agg({
                    'amount': ['count', 'mean', 'sum'],
                    'category': lambda x: x.mode()[0] if not x.empty else 'Unknown'
                }).round(2)
                
                cluster_summary.columns = ['Count', 'Avg Amount', 'Total Amount', 'Main Category']
                st.dataframe(cluster_summary)
                
                # Visualize clusters
                fig = px.scatter(
                    clusters,
                    x='amount',
                    y='days_since_start',
                    color='cluster',
                    hover_data=['description', 'category'],
                    title="Transaction Clusters"
                )
                st.plotly_chart(fig, use_container_width=True)
                
        except Exception as e:
            st.error(f"Error in clustering: {str(e)}")
        
        # Category analysis
        st.subheader("Category Analysis")
        
        expenses = category_summary[category_summary['total_amount'] < 0].copy()
        expenses['total_amount'] = expenses['total_amount'].abs()
        expenses = expenses.sort_values('total_amount', ascending=False)
        
        if not expenses.empty:
            st.dataframe(expenses.style.format({
                'total_amount': '${:,.2f}',
                'avg_amount': '${:,.2f}'
            }))
            
            # Top spending categories
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Top 5 Spending Categories")
                top_categories = expenses.head(5)
                
                fig = px.bar(
                    top_categories,
                    x='total_amount',
                    y='category',
                    orientation='h',
                    title="Highest Spending Categories"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("Transaction Frequency")
                
                fig = px.bar(
                    expenses,
                    x='category',
                    y='transaction_count',
                    title="Number of Transactions by Category"
                )
                fig.update_xaxes(tickangle=45)
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No spending data available. Please upload transaction data first.")

elif page == "Forecasting":
    st.header("Spending Forecasting")
    
    # Check if we have enough data for forecasting
    trends_data = get_spending_trends()
    
    if not trends_data.empty:
        # Show data summary
        st.subheader("Data Summary")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Categories", len(trends_data['category'].unique()))
        with col2:
            st.metric("Data Points", len(trends_data))
        with col3:
            months_of_data = len(trends_data['month'].unique())
            st.metric("Months of Data", months_of_data)
        
        # Forecasting options
        st.subheader("Forecast Settings")
        col1, col2 = st.columns(2)
        
        with col1:
            forecast_method = st.selectbox(
                "Forecasting Method",
                ["lstm", "linear", "hybrid"],
                help="LSTM: Long Short-Term Memory neural network\nLinear: Linear regression for trend analysis\nHybrid: Combines LSTM and Linear Regression"
            )
        
        with col2:
            forecast_months = st.slider("Forecast Months", 1, 6, 3)
        
        # Generate forecast
        if st.button("Generate Forecast", type="primary"):
            with st.spinner("Generating forecasts..."):
                try:
                    predictions = forecast_spending(method=forecast_method, months=forecast_months)
                    
                    if not predictions.empty:
                        st.subheader("Spending Forecast")
                        
                        # Display predictions
                        display_predictions = predictions.copy()
                        display_predictions['predicted_amount'] = display_predictions['predicted_amount'].apply(lambda x: f"${x:,.2f}")
                        display_predictions['confidence'] = display_predictions['confidence'].apply(lambda x: f"{x:.0%}")
                        
                        st.dataframe(display_predictions, use_container_width=True)
                        
                        # Visualization
                        fig = px.bar(
                            predictions,
                            x='category',
                            y='predicted_amount',
                            color='confidence',
                            title="Predicted Spending by Category",
                            color_continuous_scale='RdYlGn'
                        )
                        fig.update_xaxes(tickangle=45)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Total prediction
                        total_predicted = predictions['predicted_amount'].sum()
                        avg_confidence = predictions['confidence'].mean()
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Total Predicted", f"${total_predicted:,.2f}")
                        with col2:
                            st.metric("Average Confidence", f"{avg_confidence:.0%}")
                        
                        # Savings goals suggestion
                        st.subheader("Savings Goal Suggestion")
                        current_spending = get_summary_by_category()
                        if not current_spending.empty:
                            current_expenses = abs(current_spending[current_spending['total_amount'] < 0]['total_amount'].sum())
                            suggested_savings = max(0, (current_expenses - total_predicted) * 0.2)
                            st.info(f"Suggested savings goal: ${suggested_savings:,.2f} per month")
                    
                    else:
                        st.warning("No predictions generated. Need at least 3 months of data per category.")
                
                except Exception as e:
                    st.error(f"Error generating forecast: {str(e)}")
    else:
        st.info("Not enough transaction data for forecasting. Please upload more data first.")

elif page == "Alerts & Insights":
    st.header("Alerts & Insights")
    
    # Unusual transactions detection
    st.subheader("Unusual Transaction Detection")
    
    try:
        unusual_transactions = detect_unusual_transactions()
        
        if not unusual_transactions.empty:
            st.warning(f"Found {len(unusual_transactions)} unusual transactions!")
            
            # Display unusual transactions
            display_cols = ['date', 'description', 'amount', 'category']
            if 'z_score' in unusual_transactions.columns:
                display_cols.append('z_score')
            
            unusual_display = unusual_transactions[display_cols].copy()
            if 'z_score' in unusual_display.columns:
                unusual_display['z_score'] = unusual_display['z_score'].round(2)
            
            st.dataframe(unusual_display.style.format({
                'amount': '${:,.2f}',
                'z_score': '{:.2f}' if 'z_score' in unusual_display.columns else None
            }))
            
            # Visualization
            fig = px.scatter(
                unusual_transactions,
                x='date',
                y='amount',
                color='category',
                size='z_score' if 'z_score' in unusual_transactions.columns else None,
                hover_data=['description'],
                title="Unusual Transactions Over Time"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("No unusual transactions detected!")
    
    except Exception as e:
        st.error(f"Error detecting unusual transactions: {str(e)}")
    
    # Spending insights and alerts
    st.subheader("Spending Insights")
    
    category_summary = get_summary_by_category()
    monthly_summary = get_monthly_summary()
    
    if not category_summary.empty and not monthly_summary.empty:
        expenses = category_summary[category_summary['total_amount'] < 0].copy()
        expenses['total_amount'] = expenses['total_amount'].abs()
        
        if not expenses.empty:
            # Top spending category
            top_category = expenses.loc[expenses['total_amount'].idxmax()]
            st.info(f"Top Spending Category: {top_category['category']} (${top_category['total_amount']:,.2f})")
            
            # Monthly trend analysis
            monthly_totals = monthly_summary.groupby('month')['total_amount'].sum().reset_index()
            monthly_totals = monthly_totals.sort_values('month')
            
            if len(monthly_totals) >= 2:
                last_month = monthly_totals.iloc[-1]['total_amount']
                prev_month = monthly_totals.iloc[-2]['total_amount']
                change = ((last_month - prev_month) / abs(prev_month)) * 100
                
                # Spending alerts
                if change > 15:
                    st.error(f"Alert: Spending increased by {change:.1f}% - Consider reviewing your budget!")
                elif change > 5:
                    st.warning(f"Caution: Spending increased by {change:.1f}% from last month")
                elif change < -10:
                    st.success(f"Great job: Spending decreased by {abs(change):.1f}% from last month")
                else:
                    st.info(f"Spending change: {change:+.1f}% from last month")
            
            # Category spending alerts
            for _, category in expenses.head(5).iterrows():
                avg_transaction = category['total_amount'] / category['transaction_count']
                if avg_transaction > 100:  # Alert for high average transactions
                    st.warning(f"High average transaction in {category['category']}: ${avg_transaction:.2f}")
    else:
        st.info("Not enough data for insights. Please upload more transaction data.")