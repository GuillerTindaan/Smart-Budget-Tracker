import pandas as pd

# Enhanced category keywords to cover broad area
CATEGORY_KEYWORDS = {
    'groceries': ['grocery', 'supermarket', 'aldi', 'whole foods', 'kroger', 'safeway', 'wegmans', 'trader joe'],
    'rent': ['rent', 'landlord', 'property management', 'housing'],
    'utilities': ['electric', 'gas', 'water', 'utility', 'power', 'energy', 'internet', 'cable', 'phone'],
    'entertainment': ['netflix', 'spotify', 'cinema', 'movie', 'theater', 'gaming', 'steam', 'hulu', 'disney'],
    'food': ['restaurant', 'mcdonalds', 'pizza', 'ubereats', 'doordash', 'grubhub', 'cafe', 'starbucks'],
    'transportation': ['uber', 'lyft', 'metro', 'bus', 'gasoline', 'gas station', 'parking', 'taxi'],
    'income': ['salary', 'deposit', 'payroll', 'wages', 'bonus', 'refund'],
    'healthcare': ['pharmacy', 'doctor', 'hospital', 'medical', 'dentist', 'insurance'],
    'shopping': ['amazon', 'target', 'walmart', 'clothing', 'shoes', 'mall'],
    'education': ['tuition', 'textbook', 'school', 'course', 'training'],
    'savings': ['savings', 'investment', 'retirement', '401k', 'ira'],
    'debt': ['credit card', 'loan', 'mortgage', 'interest', 'payment']
}

def infer_category(description):
    if pd.isna(description):
        return 'other'
    
    description = str(description).lower()
    
    # Score each category based on keyword matches
    category_scores = {}
    for category, keywords in CATEGORY_KEYWORDS.items():
        score = sum(1 for keyword in keywords if keyword in description)
        if score > 0:
            category_scores[category] = score
    
    if category_scores:
        # Return the category with the highest score
        return max(category_scores, key=category_scores.get)
    
    return 'other'

def detect_recurring_transactions(df):
    df = df.copy()
    df['is_recurring'] = False
    
    # Group by description and amount to find potential recurring transactions
    for name, group in df.groupby(['description', 'amount']):
        if len(group) >= 2:
            # Check if transactions occur roughly monthly
            dates = pd.to_datetime(group['date']).sort_values()
            if len(dates) >= 2:
                diffs = dates.diff().dropna()
                # If most differences are between 25-35 days, it's likely recurring
                monthly_diffs = diffs[diffs.dt.days.between(25, 35)]
                if len(monthly_diffs) >= len(diffs) * 0.5:
                    df.loc[group.index, 'is_recurring'] = True
    
    return df

def clean_transaction_data(df):
    df = df.copy()
    
    # Standardize column names
    df.columns = [col.lower().strip() for col in df.columns]
    
    # Handle date column
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
    else:
        # If no date column, create one with today's date
        df['date'] = pd.Timestamp.now().date()
    
    # Handle amount column
    if 'amount' in df.columns:
        # Remove currency symbols and convert to numeric
        df['amount'] = df['amount'].astype(str).str.replace(r'[\$,]', '', regex=True)
        df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
    else:
        raise ValueError("Amount column is required")
    
    # Handle description column
    if 'description' not in df.columns:
        df['description'] = 'Transaction'
    
    # Handle category column
    if 'category' not in df.columns or df['category'].isna().all():
        df['category'] = df['description'].apply(infer_category)
    else:
        df['category'] = df['category'].fillna(df['description'].apply(infer_category))
    
    # Add subcategory if not present
    if 'subcategory' not in df.columns:
        df['subcategory'] = None
    
    # Add notes if not present
    if 'notes' not in df.columns:
        df['notes'] = None
    
    # Remove rows with invalid data
    df = df.dropna(subset=['date', 'amount'])
    
    # Remove duplicates
    df = df.drop_duplicates(subset=['date', 'description', 'amount'])
    
    # Detect recurring transactions
    df = detect_recurring_transactions(df)
    
    return df[['date', 'description', 'amount', 'category', 'subcategory', 'notes', 'is_recurring']]