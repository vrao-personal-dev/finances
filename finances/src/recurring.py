import sqlite3
import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity

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

# Function to identify recurring payments using sentence-level similarity
def identify_recurring_payments(df, sim_threshold: float = 0.8):
    """
    Return tuples:
        (month, section, description_1, description_2, similarity)
    where similarity ≥ `sim_threshold`.
    """
    df = df.copy()
    df["month"] = df["Date"].dt.to_period("M")
    df["section"] = df["Date"].dt.day // 10  # 3 windows per month (1-10, 11-20, 21-eom)

    recurring_payments: list[tuple] = []

    for (month, section), group in df.groupby(["month", "section"]):
        token_lists = group["Description"].str.lower().str.split().tolist()
        if len(token_lists) < 2:
            continue  # nothing to compare in this window

        # Train Word2Vec on this window’s descriptions
        model = Word2Vec(
            token_lists,
            vector_size=100,
            window=5,
            min_count=1,
            workers=4,
            epochs=30,
            sg=1,
        )

        # Helper: average word vectors to build a sentence vector
        def sent_vec(tokens: list[str]) -> np.ndarray | None:
            vecs = [model.wv[w] for w in tokens if w in model.wv]
            return np.mean(vecs, axis=0) if vecs else None

        sentence_vecs = [sent_vec(toks) for toks in token_lists]
        valid_idx = [i for i, v in enumerate(sentence_vecs) if v is not None]
        if len(valid_idx) < 2:
            continue

        mat = np.vstack([sentence_vecs[i] for i in valid_idx])
        sim_mat = cosine_similarity(mat)

        # Collect pairs that meet or exceed the similarity threshold
        for i_mat, j_mat in zip(*np.where(sim_mat >= sim_threshold)):
            if i_mat >= j_mat:  # keep each unordered pair once
                continue
            i_df, j_df = valid_idx[i_mat], valid_idx[j_mat]
            recurring_payments.append(
                (
                    str(month),
                    int(section),
                    group.iloc[i_df]["Description"],
                    group.iloc[j_df]["Description"],
                    float(sim_mat[i_mat, j_mat]),
                )
            )

    return recurring_payments

# Identify recurring payments
recurring_payments = identify_recurring_payments(transactions_df)

# Print results
for payment in recurring_payments:
    print(
        f"Month: {payment[0]}, Section: {payment[1]}, "
        f"Desc1: {payment[2]}, Desc2: {payment[3]}, Similarity: {payment[4]:.2f}"
    )

# Close the database connection
conn.close()

