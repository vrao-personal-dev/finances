import os
import random
import re

from db import get_conn

import fasttext.util
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, roc_curve
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.utils import resample

import numpy as np
import pandas as pd
from scipy.sparse import vstack

fasttext.util.download_model("en", if_exists="ignore")
ft = fasttext.load_model("cc.en.300.bin")

# tfidf vectorizer --> fit under data pull
# word2weight will come from the fit tfidf below
tfidf = TfidfVectorizer(analyzer="word", use_idf=True)

# 3gram Vectorizer --> fit under data pull
tfidf3gram = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 3))

LABELS_FILES = ["pairs_to_label.csv", "more_pairs_labels.csv",
"even_more_pairs_labels.csv"]

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
query = f"""
SELECT
   AccountName,
   Amount,
   COALESCE(Description, OriginalDescription) AS Description,
   COALESCE(OriginalDescription, Description) AS OriginalDescription,
   Date
FROM transactions
WHERE TransactionType = 'debit'
AND AccountName in ({placeholders})
"""

def clean_desc(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()

    # 1. \bin\b  -> Matches "in" as a whole word (avoids matching "inside")
    # 2. &       -> Matches literal &
    # 3. #[0-9]* -> Matches # followed by digits (no boundary needed before #)
    noise = r"\bin\b|&|#[0-9]*"

    text = re.sub(noise, "", text)
    return re.sub(r"\s+", " ", text).strip()

conn = get_conn()
cursor = conn.cursor()
txn_df = pd.read_sql_query(query, conn, params=SUPPORTED_ACCOUNTS)
# token_lists = txn_df["Description"].apply(clean_desc).str.lower().str.split().tolist()
token_lists = txn_df["OriginalDescription"].apply(clean_desc).str.lower().str.split().tolist()

raw_descriptions = txn_df["Description"].tolist()
# used to get idx given descriptions for labeled df
text_to_idx = {text: i for i, text in enumerate(raw_descriptions)}

def sent_vec_fasttext(tok_list: list[str]) -> np.ndarray:
    return np.mean([ft.get_word_vector(t) for t in tok_list], axis=0)

# just fasttext- no weighting- gives us >0.9 (1.4%), <0.2` (64%), 0.2 - 0.4`
# (31.3%), 0.4 - 0.7` (3%)
vecs = np.vstack([sent_vec_fasttext(t) for t in token_lists])
sim_mat = cosine_similarity(vecs)

def get_bucket_counts(sim_mat):
    upper_triangle_indices = np.triu_indices_from(sim_mat, k=1)
    scores = sim_mat[upper_triangle_indices]
    count_buckets_1 = np.sum(scores >= 0.9)
    count_buckets_2 = np.sum((scores >= 0.2) & (scores < 0.4))
    count_buckets_3 = np.sum((scores >= 0.4) & (scores < 0.7))
    count_buckets_0 = np.sum(scores < 0.2)

    tot = len(sim_mat) * len(sim_mat) / 2 - len(sim_mat) / 2.0
    print("Bucket percentages for >=0.9, 0.2-0.4, 0.4-0.7, <0.2")
    for buc in [count_buckets_1, count_buckets_2, count_buckets_3, count_buckets_0]:
        print((buc / tot * 100).round(4), end=' ')
    print()

def generate_review_df(sim_mat, min_sim=0.7, max_sim=0.8):
    u = np.triu_indices_from(sim_mat, k=1)
    scores = sim_mat[iu]
    idx1_all = iu[0]
    idx2_all = iu[1]

    print(f"Finding pairs with similarity between {min_sim} and {max_sim}...")
    mask = (scores >= min_sim) & (scores < max_sim)

    filtered_scores = scores[mask]
    filtered_idx1 = idx1_all[mask]
    filtered_idx2 = idx2_all[mask]

    # 4. Sort by similarity to see the strongest pairs first
    sort_order = np.argsort(filtered_scores)[::-1]

    # 5. Build a DataFrame for easy viewing
    review_df = pd.DataFrame(
        {
            "similarity": filtered_scores[sort_order],
            "desc1": [descriptions[i] for i in filtered_idx1[sort_order]],
            "desc2": [descriptions[i] for i in filtered_idx2[sort_order]],
        }
    )


def get_models(token_list):
    descs = [" ".join(tok_list) for tok_list in token_list]
    tfidf.fit(descs)
    word2weight = dict(zip(tfidf.get_feature_names_out(), tfidf.idf_))
    tfidf3gram.fit(descs)
    return word2weight, tfidf3gram

def sent_vec_weighted(tok_list):
    vectors = []
    weights = []
    for t in tok_list:
        v = ft.get_word_vector(t)
        w = word2weight.get(t.lower(), 0.5)
        vectors.append(v)
        weights.append(w)
    return np.average(vectors, axis=0, weights=weights)

word2weight, tfidf3gram = get_models(token_lists)  # needed for weighting below
# fastext w/ tfidf weights gives us >0.9 (1.3%), <0.2` (64.5%), 0.2 - 0.4`
# (30.9%), 0.4 - 0.7` (3.0%)
vecs_weighted = np.vstack([sent_vec_weighted(t) for t in token_lists])
sim_mat_weighted = cosine_similarity(vecs_weighted)

def three_gram_vec_weighted(tok_list: list[str]) -> np.ndarray:
    return tfidf3gram.transform([" ".join(tok_list)])

# tfidf3gram gives us >0.9 (1.2%), <0.2 (97.7%), 0.2-0.4
# (0.4%), 0.4 - 0.7 (0.6%)
vecs_three_gram = vstack([three_gram_vec_weighted(t) for t in token_lists])
sim_mat_3gram = cosine_similarity(vecs_three_gram)

def get_score_by_idx(sim_mat, idx1, idx2):
    try:
        i = int(idx1)
        j = int(idx2)
        if i < sim_mat.shape[0] and j < sim_mat.shape[0]:
            return sim_mat[i, j]
        return 0.0
    except (ValueError, IndexError):
        return 0.0


def get_score_by_text(sim_mat, t1, t2):
    try:
        i = text_to_idx[t1]
        j = text_to_idx[t2]
        return sim_mat[i, j]
    except KeyError:
        return 0.0


def run_bootstrap(df, n_iterations=1000):
    thresholds_boot = []
    # Generate seeds
    seeds = {42}
    while len(seeds) < n_iterations:
        seeds.add(random.randint(0, 1_000_000))
    seed_list = list(seeds)
    random.shuffle(seed_list)

    for seed in seed_list:
        boot_df = resample(df, replace=True, n_samples=len(df), random_state=seed)
        if boot_df["is_match"].nunique() < 2:
            continue

        fpr, tpr, thresh = roc_curve(boot_df["is_match"], boot_df["score"])
        optimal_idx = np.argmax(tpr - fpr)
        thresholds_boot.append(thresh[optimal_idx])

    lower_ci = np.percentile(thresholds_boot, 2.5)
    upper_ci = np.percentile(thresholds_boot, 97.5)
    return np.mean(thresholds_boot), lower_ci, upper_ci


def acc_metrics(labeled_df, thresh):
    y_true = labeled_df["is_match"]
    y_pred = labeled_df["score"] >= thresh
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

    return tpr, tnr, precision


def main():
    # Load labeled data
    labeled_dfs = []
    for fname in LABELS_FILES:
        if os.path.exists(fname):
            labeled_dfs.append(pd.read_csv(fname))

    if not labeled_dfs:
        print("No labeled data found (pairs_to_label.csv, etc). Exiting.")
        return

    full_labeled_df = pd.concat(labeled_dfs, ignore_index=True)
    print(f"Loaded {len(full_labeled_df)} labeled pairs.")

    # ---------------------------------------------------------
    # Analysis 1: Using 'Description' (Coalesced)
    # ---------------------------------------------------------
    # Score the labeled data
    labeled_df_desc = full_labeled_df.copy()
    labeled_df_desc["score"] = labeled_df_desc.apply(
        lambda x: get_score_by_text(sim_mat, x["desc1_text"], x["desc2_text"]), axis=1
    )
    labeled_df_weighted = full_labeled_df.copy()
    labeled_df_weighted["score"] = labeled_df_weighted.apply(
        lambda x: get_score_by_text(sim_mat_weighted, x["desc1_text"], x["desc2_text"]),
        axis=1,
    )
    labeled_df_3gram = full_labeled_df.copy()
    labeled_df_3gram["score"] = labeled_df_3gram.apply(
        lambda x: get_score_by_text(sim_mat_3gram, x["desc1_text"], x["desc2_text"]),
        axis=1,
    )

    print("\n--- FastText (unweighted) ---")
    mean_desc, low_desc, high_desc = run_bootstrap(labeled_df_desc)
    get_bucket_counts(sim_mat)
    print(f"Optimal Threshold: {mean_desc:.4f}")
    print(f"95% CI:           [{low_desc:.4f}, {high_desc:.4f}]")

    # Metrics
    tpr, tnr, prec = acc_metrics(labeled_df_desc, mean_desc)
    print(f"Recall (TPR):      {tpr:.2%}")
    print(f"Specificity (TNR): {tnr:.2%}")
    print(f"Precision:         {prec:.2%}")

    print("\n--- TF-IDF Weighted FastText ---")
    mean_weighted, low_weighted, high_weighted = run_bootstrap(labeled_df_weighted)
    get_bucket_counts(sim_mat_weighted)
    print(f"Optimal Threshold: {mean_weighted:.4f}")
    print(f"95% CI:           [{low_weighted:.4f}, {high_weighted:.4f}]")

    tpr_w, tnr_w, prec_w = acc_metrics(labeled_df_weighted, mean_weighted)
    print(f"Recall (TPR):      {tpr_w:.2%}")
    print(f"Specificity (TNR): {tnr_w:.2%}")
    print(f"Precision:         {prec_w:.2%}")

    print("\n--- 3-gram TF-IDF ---")
    mean_3gram, low_3gram, high_3gram = run_bootstrap(labeled_df_3gram)
    get_bucket_counts(sim_mat_3gram)
    print(f"Optimal Threshold: {mean_3gram:.4f}")
    print(f"95% CI:           [{low_3gram:.4f}, {high_3gram:.4f}]")

    tpr_3g, tnr_3g, prec_3g = acc_metrics(labeled_df_3gram, mean_3gram)
    print(f"Recall (TPR):      {tpr_3g:.2%}")
    print(f"Specificity (TNR): {tnr_3g:.2%}")
    print(f"Precision:         {prec_3g:.2%}")


if __name__ == "__main__":
    main()
