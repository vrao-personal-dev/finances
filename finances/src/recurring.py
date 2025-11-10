import sqlite3
import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict

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

# Function to identify recurring payments across DIFFERENT periods
def identify_recurring_payments(
    df: pd.DataFrame,
    sim_threshold: float = 0.8,
    min_periods: int = 3,
) -> list[dict]:
    """
    Detect subscription-like recurring payments.

    A description is considered *recurring* when **similar transactions appear
    in at least `min_periods` distinct month-sections** (3 per month).

    Returns
    -------
    List[dict] – each dict has
        description : str
        periods     : set[str]   (e.g. {"2025-01-0", "2025-02-2", …})
        count       : int        (# distinct periods)
    """
    df = df.copy()
    df["month"] = df["Date"].dt.to_period("M")
    df["section"] = df["Date"].dt.day // 10  # 0,1,2  →  1-10,11-20,21-eom
    df["period_id"] = df["month"].astype(str) + "-" + df["section"].astype(str)

    # ── 1️⃣  Build a Word2Vec model on *all* descriptions ────────────────────
    token_lists = df["Description"].str.lower().str.split().tolist()
    w2v_model = Word2Vec(
        token_lists,
        vector_size=100,
        window=5,
        min_count=1,
        workers=4,
        epochs=30,
        sg=1,
    )

    # ── 2️⃣  Convert every description to a sentence vector ─────────────────
    def sent_vec(tokens: list[str]) -> np.ndarray | None:
        vecs = [w2v_model.wv[w] for w in tokens if w in w2v_model.wv]
        return np.mean(vecs, axis=0) if vecs else None

    sentence_vecs = [sent_vec(toks) for toks in token_lists]
    valid_idx = [i for i, v in enumerate(sentence_vecs) if v is not None]

    if len(valid_idx) < 2:
        return []

    vec_matrix = np.vstack([sentence_vecs[i] for i in valid_idx])
    sim_mat = cosine_similarity(vec_matrix)

    # ── 3️⃣  Accumulate periods for every vector that is similar to another
    #         vector *in a different period* ────────────────────────────────
    period_ids = df.loc[valid_idx, "period_id"].to_numpy(str)
    desc_texts = df.loc[valid_idx, "Description"].to_numpy(str)

    recurring_map: defaultdict[int, set[str]] = defaultdict(set)

    for i_mat, j_mat in zip(*np.where(sim_mat >= sim_threshold)):
        if i_mat == j_mat:
            continue  # skip self-pair
        if period_ids[i_mat] == period_ids[j_mat]:
            continue  # same period → ignore
        recurring_map[i_mat].add(period_ids[i_mat])
        recurring_map[i_mat].add(period_ids[j_mat])
        recurring_map[j_mat].add(period_ids[i_mat])
        recurring_map[j_mat].add(period_ids[j_mat])

    # ── 4️⃣  Build result list ──────────────────────────────────────────────
    results: list[dict] = []
    for idx, periods in recurring_map.items():
        if len(periods) >= min_periods:
            results.append(
                {
                    "description": desc_texts[idx],
                    "periods": periods,
                    "count": len(periods),
                }
            )
    return results

# Identify recurring payments (≥3 periods by default)
recurring_payments = identify_recurring_payments(transactions_df)

# Print summary
for rec in recurring_payments:
    periods_sorted = ", ".join(sorted(rec["periods"]))
    print(
        f"[{rec['count']:2d} periods] {rec['description']}\n"
        f"    periods → {periods_sorted}\n"
    )

# Close the database connection
conn.close()

