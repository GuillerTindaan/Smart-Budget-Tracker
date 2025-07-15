import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from database import get_spending_trends, get_all_transactions

def linear_regression_forecast(category_data, months=1):
    if len(category_data) < 3:
        return None, 0.2
    
    # Prepare data
    category_data = category_data.copy()
    category_data['month_num'] = range(len(category_data))
    
    # Fit linear regression
    X = category_data[['month_num']]
    y = category_data['amount'].abs()
    
    model = LinearRegression()
    model.fit(X, y)
    
    # Predict next months
    predictions = []
    for i in range(months):
        next_month = len(category_data) + i
        prediction = model.predict([[next_month]])[0]
        predictions.append(prediction)
    
    # Calculate confidence based on R-squared and prediction stability
    r_squared = model.score(X, y)
    confidence = max(0.2, min(0.8, r_squared))
    
    return predictions, confidence

def lstm_forecast(category_data, months=1):
    
    if len(category_data) < 6:
        return None, 0.1
    
    try:
        # Prepare data
        amounts = category_data['amount'].abs().values
        
        # Scale data
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(amounts.reshape(-1, 1))
        
        # Create sequences
        sequence_length = min(3, len(amounts) - 2)
        X, y = [], []
        
        for i in range(sequence_length, len(scaled_data)):
            X.append(scaled_data[i-sequence_length:i, 0])
            y.append(scaled_data[i, 0])
        
        if len(X) < 3:
            return None, 0.1
        
        X = np.array(X)
        y = np.array(y)
        
        # Reshape for LSTM
        X = X.reshape((X.shape[0], X.shape[1], 1))
        
        # Build LSTM model
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(sequence_length, 1)),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mse')
        
        # Train model
        model.fit(X, y, epochs=50, batch_size=1, verbose=0)
        
        # Generate predictions
        predictions = []
        current_sequence = scaled_data[-sequence_length:].reshape(1, sequence_length, 1)
        
        for _ in range(months):
            prediction_scaled = model.predict(current_sequence, verbose=0)
            prediction = scaler.inverse_transform(prediction_scaled)[0, 0]
            predictions.append(prediction)
            
            # Update sequence for next prediction
            current_sequence = np.append(current_sequence[:, 1:, :], 
                                       prediction_scaled.reshape(1, 1, 1), axis=1)
        
        # Calculate confidence based on model performance
        train_predictions = model.predict(X, verbose=0)
        train_predictions = scaler.inverse_transform(train_predictions)
        actual_values = scaler.inverse_transform(y.reshape(-1, 1))
        
        mse = mean_squared_error(actual_values, train_predictions)
        confidence = max(0.3, min(0.9, 1 - (mse / np.var(actual_values))))
        
        return predictions, confidence
        
    except Exception as e:
        return None, 0.1

def hybrid_forecast(category_data, months=1):
    lstm_pred, lstm_conf = lstm_forecast(category_data, months)
    lr_pred, lr_conf = linear_regression_forecast(category_data, months)
    
    if lstm_pred is not None and lr_pred is not None:
        # Weighted combination based on confidence
        total_conf = lstm_conf + lr_conf
        if total_conf > 0:
            lstm_weight = lstm_conf / total_conf
            lr_weight = lr_conf / total_conf
            
            combined_pred = [lstm_weight * l + lr_weight * r 
                           for l, r in zip(lstm_pred, lr_pred)]
            combined_conf = min(0.9, (lstm_conf + lr_conf) / 2)
            
            return combined_pred, combined_conf
    
    # Fallback to the best available method
    if lstm_pred is not None:
        return lstm_pred, lstm_conf
    elif lr_pred is not None:
        return lr_pred, lr_conf
    else:
        return None, 0.1

def cluster_transactions(user_id=1):
    try:
        # Get all transactions
        df = get_all_transactions(user_id)
        
        if df.empty or len(df) < 10:
            return pd.DataFrame()
        
        # Prepare features for clustering
        df['amount_abs'] = df['amount'].abs()
        df['date'] = pd.to_datetime(df['date'])
        df['days_since_start'] = (df['date'] - df['date'].min()).dt.days
        df['month'] = df['date'].dt.month
        df['day_of_week'] = df['date'].dt.dayofweek
        
        # Select features for clustering
        features = ['amount_abs', 'days_since_start', 'month', 'day_of_week']
        X = df[features].fillna(0)
        
        # Determine optimal number of clusters (3-8 based on data size)
        n_clusters = min(8, max(3, len(df) // 20))
        
        # Perform K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        df['cluster'] = kmeans.fit_predict(X)
        
        return df[['date', 'description', 'amount', 'category', 'cluster', 'days_since_start']]
        
    except Exception as e:
        return pd.DataFrame()

def forecast_spending(user_id=1, method='lstm', months=1):
    # Get spending trends data
    df = get_spending_trends(user_id, months=12)
    
    if df.empty:
        return pd.DataFrame()
    
    # Process data
    df["month"] = pd.to_datetime(df["month"])
    df = df.sort_values("month")
    df['amount'] = df['amount'].abs()
    
    predictions = []
    
    for category in df["category"].unique():
        cat_df = df[df["category"] == category].sort_values("month")
        
        # Need at least 3 months of data for meaningful prediction
        if len(cat_df) < 3:
            continue
        
        prediction = None
        confidence = 0.2
        method_used = method
        
        # Apply selected forecasting method
        if method == 'lstm':
            prediction, confidence = lstm_forecast(cat_df, months)
            if prediction is None:
                # Fallback to linear regression
                prediction, confidence = linear_regression_forecast(cat_df, months)
                method_used = 'linear_fallback'
        
        elif method == 'linear':
            prediction, confidence = linear_regression_forecast(cat_df, months)
        
        elif method == 'hybrid':
            prediction, confidence = hybrid_forecast(cat_df, months)
            method_used = 'hybrid'
        
        # Validate and cap predictions
        if prediction is not None:
            # Use the first month's prediction for display
            first_month_pred = prediction[0]
            
            # Cap at 3x historical maximum
            max_historical = cat_df['amount'].max()
            first_month_pred = min(first_month_pred, max_historical * 3)
            
            # Ensure positive prediction
            if first_month_pred > 0:
                predictions.append({
                    "category": category,
                    "predicted_amount": round(first_month_pred, 2),
                    "confidence": round(confidence, 2),
                    "method": method_used
                })
    
    return pd.DataFrame(predictions)

def suggest_savings_goal(user_id=1):
    try:
        # Get current spending data
        current_spending = get_spending_trends(user_id, months=3)
        
        if current_spending.empty:
            return None
        
        # Calculate average monthly spending
        monthly_spending = current_spending.groupby('month')['amount'].sum().abs()
        avg_monthly_spending = monthly_spending.mean()
        
        # Get spending trend
        forecast_data = forecast_spending(user_id, method='linear', months=3)
        
        if not forecast_data.empty:
            predicted_spending = forecast_data['predicted_amount'].sum()
            
            # Calculate potential savings
            if predicted_spending < avg_monthly_spending:
                potential_savings = avg_monthly_spending - predicted_spending
                suggested_goal = potential_savings * 0.8  # 80% of potential savings
            else:
                suggested_goal = avg_monthly_spending * 0.1  # 10% of current spending
            
            return {
                'suggested_amount': round(suggested_goal, 2),
                'current_spending': round(avg_monthly_spending, 2),
                'predicted_spending': round(predicted_spending, 2)
            }
        
        return None
        
    except Exception as e:
        return None