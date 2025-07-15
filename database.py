import sqlite3
import pandas as pd
import numpy as np
import os

DB_NAME = 'database.db'

def get_connection():
    return sqlite3.connect(DB_NAME)

def check_column_exists(table_name, column_name):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(f"PRAGMA table_info({table_name})")
    columns = [column[1] for column in cursor.fetchall()]
    conn.close()
    return column_name in columns

def initialize_database():
    conn = get_connection()
    cursor = conn.cursor()
    
    # Check if this is a fresh database or needs migration
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='transactions'")
    table_exists = cursor.fetchone() is not None
    
    # Check if default user already exists
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='users'")
    users_table_exists = cursor.fetchone() is not None
    
    # Users table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Only insert default user if users table is empty
    if users_table_exists:
        cursor.execute("SELECT COUNT(*) FROM users")
        user_count = cursor.fetchone()[0]
        if user_count == 0:
            cursor.execute("INSERT INTO users (name, email) VALUES (?, ?)", ("Default User", "user@example.com"))
    else:
        cursor.execute("INSERT INTO users (name, email) VALUES (?, ?)", ("Default User", "user@example.com"))
    
    # Enhanced transactions table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS transactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER DEFAULT 1,
            date DATE NOT NULL,
            description TEXT NOT NULL,
            amount REAL NOT NULL,
            category TEXT NOT NULL,
            subcategory TEXT,
            notes TEXT,
            is_recurring BOOLEAN DEFAULT 0,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    # If table existed before, check for missing columns and add them
    if table_exists:
        required_columns = {
            'user_id': 'INTEGER DEFAULT 1',
            'subcategory': 'TEXT',
            'notes': 'TEXT',
            'is_recurring': 'BOOLEAN DEFAULT 0'
        }
        
        for col_name, col_type in required_columns.items():
            if not check_column_exists('transactions', col_name):
                print(f"Adding missing column: {col_name}")
                cursor.execute(f"ALTER TABLE transactions ADD COLUMN {col_name} {col_type}")
    
    # Categories table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS categories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            parent_category TEXT,
            budget_limit REAL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Goals table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS goals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER DEFAULT 1,
            goal_type TEXT NOT NULL,
            target_amount REAL NOT NULL,
            current_amount REAL DEFAULT 0,
            target_date DATE,
            description TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    # Budgets table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS budgets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER DEFAULT 1,
            category TEXT NOT NULL,
            monthly_limit REAL NOT NULL,
            current_spent REAL DEFAULT 0,
            month_year TEXT NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    conn.commit()
    conn.close()

def insert_transactions(df, user_id=1):
    conn = get_connection()
    
    # Ensure user_id column exists
    if not check_column_exists('transactions', 'user_id'):
        cursor = conn.cursor()
        cursor.execute("ALTER TABLE transactions ADD COLUMN user_id INTEGER DEFAULT 1")
        conn.commit()
    
    # Check for duplicates before inserting
    try:
        existing_query = """
            SELECT date, description, amount 
            FROM transactions 
            WHERE user_id = ?
        """
        existing_df = pd.read_sql(existing_query, conn, params=[user_id])
    except:
        # If user_id doesn't exist, create a fallback query
        existing_query = """
            SELECT date, description, amount 
            FROM transactions
        """
        existing_df = pd.read_sql(existing_query, conn)
    
    if not existing_df.empty:
        # Create a composite key for duplicate detection
        df['composite_key'] = df['date'].astype(str) + '_' + df['description'] + '_' + df['amount'].astype(str)
        existing_df['composite_key'] = existing_df['date'].astype(str) + '_' + existing_df['description'] + '_' + existing_df['amount'].astype(str)
        
        # Filter out duplicates
        df = df[~df['composite_key'].isin(existing_df['composite_key'])]
        df = df.drop('composite_key', axis=1)
    
    if not df.empty:
        df['user_id'] = user_id
        df.to_sql('transactions', conn, if_exists='append', index=False)
    
    conn.close()
    return len(df)

def get_all_transactions(user_id=1):
    conn = get_connection()
    
    # Check if user_id column exists
    if check_column_exists('transactions', 'user_id'):
        query = """
            SELECT * FROM transactions 
            WHERE user_id = ?
            ORDER BY date DESC
        """
        params = [user_id]
    else:
        # Fallback for tables without user_id
        query = """
            SELECT * FROM transactions 
            ORDER BY date DESC
        """
        params = []
    
    result = pd.read_sql(query, conn, params=params)
    conn.close()
    return result

def get_summary_by_category(user_id=1, date_range=None):
    conn = get_connection()
    
    # Check if user_id column exists
    if check_column_exists('transactions', 'user_id'):
        base_query = """
            SELECT category, SUM(amount) as total_amount, COUNT(*) as transaction_count,
                   AVG(amount) as avg_amount, MIN(date) as first_transaction, MAX(date) as last_transaction
            FROM transactions
            WHERE user_id = ?
        """
        params = [user_id]
    else:
        # Fallback for tables without user_id
        base_query = """
            SELECT category, SUM(amount) as total_amount, COUNT(*) as transaction_count,
                   AVG(amount) as avg_amount, MIN(date) as first_transaction, MAX(date) as last_transaction
            FROM transactions
            WHERE 1 = 1
        """
        params = []
    
    if date_range:
        base_query += " AND date BETWEEN ? AND ?"
        params.extend(date_range)
    
    base_query += " GROUP BY category ORDER BY total_amount DESC"
    
    result = pd.read_sql(base_query, conn, params=params)
    conn.close()
    return result

def get_monthly_summary(user_id=1):
    conn = get_connection()
    
    if check_column_exists('transactions', 'user_id'):
        query = """
            SELECT strftime('%Y-%m', date) as month, category, 
                   SUM(amount) as total_amount,
                   COUNT(*) as transaction_count,
                   AVG(amount) as avg_amount
            FROM transactions
            WHERE user_id = ?
            GROUP BY month, category
            ORDER BY month, category
        """
        params = [user_id]
    else:
        query = """
            SELECT strftime('%Y-%m', date) as month, category, 
                   SUM(amount) as total_amount,
                   COUNT(*) as transaction_count,
                   AVG(amount) as avg_amount
            FROM transactions
            GROUP BY month, category
            ORDER BY month, category
        """
        params = []
    
    result = pd.read_sql(query, conn, params=params)
    conn.close()
    return result

def get_spending_trends(user_id=1, months=12):
    conn = get_connection()
    
    if check_column_exists('transactions', 'user_id'):
        query = """
            SELECT strftime('%Y-%m', date) as month, category, SUM(amount) as amount
            FROM transactions
            WHERE user_id = ? AND date >= date('now', '-{} months')
            GROUP BY month, category
            ORDER BY month
        """.format(months)
        params = [user_id]
    else:
        query = """
            SELECT strftime('%Y-%m', date) as month, category, SUM(amount) as amount
            FROM transactions
            WHERE date >= date('now', '-{} months')
            GROUP BY month, category
            ORDER BY month
        """.format(months)
        params = []
    
    result = pd.read_sql(query, conn, params=params)
    conn.close()
    return result

def detect_unusual_transactions(user_id=1, threshold=2.0):
    conn = get_connection()
    
    if check_column_exists('transactions', 'user_id'):
        query = """
            SELECT *, 
                   (SELECT AVG(amount) FROM transactions t2 WHERE t2.category = t1.category AND t2.user_id = ?) as avg_amount,
                   (SELECT COUNT(*) FROM transactions t2 WHERE t2.category = t1.category AND t2.user_id = ?) as category_count
            FROM transactions t1
            WHERE user_id = ?
            ORDER BY date DESC
        """
        params = [user_id, user_id, user_id]
    else:
        query = """
            SELECT *, 
                   (SELECT AVG(amount) FROM transactions t2 WHERE t2.category = t1.category) as avg_amount,
                   (SELECT COUNT(*) FROM transactions t2 WHERE t2.category = t1.category) as category_count
            FROM transactions t1
            ORDER BY date DESC
        """
        params = []
    
    df = pd.read_sql(query, conn, params=params)
    conn.close()
    
    if df.empty:
        return pd.DataFrame()
    
    # Calculate z-scores for each category
    unusual_transactions = []
    for category in df['category'].unique():
        cat_df = df[df['category'] == category]
        if len(cat_df) > 1:
            mean_amount = cat_df['amount'].mean()
            std_amount = cat_df['amount'].std()
            
            if std_amount > 0:
                cat_df['z_score'] = np.abs((cat_df['amount'] - mean_amount) / std_amount)
                unusual = cat_df[cat_df['z_score'] > threshold]
                unusual_transactions.append(unusual)
    
    if unusual_transactions:
        return pd.concat(unusual_transactions, ignore_index=True)
    else:
        return pd.DataFrame()