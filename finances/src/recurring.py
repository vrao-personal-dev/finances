import sqlite3
import pandas as pd
from gensim.models import Word2Vec
from datetime import timedelta

# Connect to the database
conn = sqlite3.connect('finances/src/mint_transactions.db')  # double check path
cursor = conn.cursor()

# Supported accounts we want to consider for recurring-payment detection
SUPPORTED_ACCOUNTS = (
    "Savings",
    "SOFI Credit Card",
    "SOFI Checking",
    "SOFI Saving",
    "Apple Card",
    "Apple Cash",
    "Venmo",
    "Robinhood Credit Card",
)
placeholders = ",".join(["?"] * len(SUPPORTED_ACCOUNTS))

# Load transactions into a DataFrame
query = f"""
    SELECT
        AccountName,
        Amount,
        COALESCE(Description, OriginalDescription) AS Description,
        Date
    FROM transactions
    WHERE TransactionType = 'debit'
      AND AccountName IN ({placeholders})
"""
transactions_df = pd.read_sql_query(query, conn, params=SUPPORTED_ACCOUNTS)

# Convert Date column to Datetime
transactions_df['Date'] = pd.to_datetime(transactions_df['Date'])

# Function to identify recurring payments
def identify_recurring_payments(df):
    # Group by month and section
    df['month'] = df['Date'].dt.to_period('M')
    df['section'] = df['Date'].dt.day // 10  # Divide month into 3 sections

    recurring_payments = []

    for (month, section), group in df.groupby(['month', 'section']):
        # Prepare descriptions for Word2Vec
        descriptions = group['Description'].apply(lambda x: x.split()).tolist()

        # Train Word2Vec model
        model = Word2Vec(descriptions, vector_size=100, window=5, min_count=1, workers=4)

        # Find similar descriptions
        for desc in descriptions:
            # `desc` is already a list of tokens → pass directly
            similar_descs = model.wv.most_similar(positive=desc, topn=5)
            for similar_desc, similarity in similar_descs:
                if similarity > 0.8:  # Threshold for similarity
                    recurring_payments.append((month, section, ' '.join(desc), ' '.join(similar_desc)))

    return recurring_payments

# Identify recurring payments
recurring_payments = identify_recurring_payments(transactions_df)

# Print results
for payment in recurring_payments:
    print(f"Month: {payment[0]}, Section: {payment[1]}, Description: {payment[2]}, Similar Description: {payment[3]}")

# Close the database connection
conn.close()

