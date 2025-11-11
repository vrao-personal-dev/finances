import sqlite3
import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import json

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

# Function to convert period + section to int
def _period_to_int(period_id: str) -> int:
    """ period looks like 2025-02-<section> sec = day // 10"""
    year, month, section = period_id.split("-")
    return int(year) * 36 + (int(month) - 1) * 3 + int(section)

# Function to identify recurring payments across DIFFERENT periods
def identify_recurring_payments(
    df: pd.DataFrame,
    sim_threshold: float = 0.8,
    min_periods: int = 3,
    min_consec: int = 3,
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
    with open("sentence_vecs.json", "w") as outfile: json.dump([vec.tolist()
for vec in sentence_vecs],
outfile)
    valid_idx = [i for i, v in enumerate(sentence_vecs) if v is not None]

    if len(valid_idx) < 2:
        return []

    vec_matrix = np.vstack([sentence_vecs[i] for i in valid_idx])
    sim_mat = cosine_similarity(vec_matrix)

    # ── 3️⃣  Accumulate periods for every vector that is similar to another
    #         vector *in a different period* ────────────────────────────────
    month_ids = df.loc[valid_idx, "month"].to_numpy(str)
    desc_texts = df.loc[valid_idx, "Description"].to_numpy(str)

    recurring_map: defaultdict[int, set[str]] = defaultdict(set)

    for i_mat, j_mat in zip(*np.where(sim_mat >= sim_threshold)):
        if i_mat == j_mat:
            continue  # skip self-pair
        if month_ids[i_mat] == month_ids[j_mat]:
            continue  # same month → ignore
        recurring_map[i_mat].add(month_ids[j_mat])
        recurring_map[j_mat].add(month_ids[i_mat])

    # ── 4️⃣  Build result list ──────────────────────────────────────────────
    results: list[dict] = []
    for idx, months in recurring_map.items():
        months = list({*months, month_ids[idx]})  # include the seed's own month
        if len(months) >= min_periods and len(months) == len(set(months)):  #filter on consecutive periods and max 1 txn per month
            results.append(
                {
                    "description": desc_texts[idx],
                    "months": set(months),
                    "count": len(months),
                }
            )
    return results

# Identify recurring payments (≥3 periods by default)
recurring_payments = identify_recurring_payments(transactions_df)

# Print summary
for rec in recurring_payments:
    periods_sorted = ", ".join(sorted(rec["months"]))
    print(
        f"[{rec['count']:2d} periods] {rec['description']}\n"
        f"    periods → {periods_sorted}\n"
    )

# Close the database connection
conn.close()

