from argparse import ArgumentParser
import pandas as pd
from db import get_conn, insert_transactions, Transaction
import os
from constants import MONARCH_ACCOUNT_NAME_MAPPING  # Import the mapping
from datetime import datetime

# Default CSV download location – used when --csv-path arg is not provided
DEFAULT_CSV_PATH = "~/Downloads/transactions-192069774303065161-52d7a35a-18f1-4fca-b8e0-906d5dd23e67.csv"


def main(
    csv_path: str | None = None,
    dry_run: bool = True,
    start_date: str | None = None,
    end_date: str | None = None,
):
    # Load the Monarch CSV file
    # Resolve CSV path – CLI flag overrides default
    if csv_path is None:
        csv_path = DEFAULT_CSV_PATH
    csv_path = os.path.expanduser(csv_path)
    monarch_df = pd.read_csv(csv_path)

    latest_dates = {}

    with get_conn() as conn:
        cursor = conn.cursor()

        # Retrieve the latest transaction date for each account
        for row in cursor.execute(
            "SELECT AccountName, MAX(Date) as LatestDate FROM transactions GROUP BY AccountName"
        ):
            account_name, latest_date = row
            latest_dates[account_name] = latest_date

        # Apply the account name mapping and filter the DataFrame to include only new transactions
        # Apply account name mapping
        monarch_df["Account"] = monarch_df["Account"].map(MONARCH_ACCOUNT_NAME_MAPPING)

        # Convert Date column to Timestamp for robust comparisons
        monarch_df["Date"] = pd.to_datetime(monarch_df["Date"])

        # Convert latest known dates to Timestamp as well
        latest_dates_dt = {
            acct: pd.to_datetime(dt) if dt is not None else pd.Timestamp.min
            for acct, dt in latest_dates.items()
        }

        # Only keep rows newer than the latest date on record for each account
        mask = monarch_df["Date"] > monarch_df["Account"].map(
            latest_dates_dt
        ).fillna(pd.Timestamp.min)

        # Optional start / end date bounds
        if start_date:
            mask &= monarch_df["Date"] >= pd.to_datetime(start_date)
        if end_date:
            mask &= monarch_df["Date"] <= pd.to_datetime(end_date)

        filtered_df = monarch_df[mask]

        # Dry-run: display transactions that would be inserted
        if dry_run:
            print(
                "Dry-run mode enabled. Records dumped to your downloads folder for review:"
            )
            filtered_df.to_csv(
                os.path.expanduser("~/Downloads/filtered_monarch_txns.csv")
            )
        else:
            # Insert the filtered data into the SQLite database
            transactions = [
                Transaction(
                    Date=row["Date"].strftime("%Y-%m-%d"),
                    Description=row["Merchant"],
                    OriginalDescription=row["Original Statement"],
                    Amount=abs(row["Amount"]),
                    TransactionType="credit" if row["Amount"] > 0 else "debit",
                    Category=row["Category"],
                    AccountName=row["Account"],
                    Labels=row.get("Tags", ""),
                    Notes=row.get("Notes", ""),
                )
                for _, row in filtered_df.iterrows()
            ]
            insert_transactions(transactions)


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Write Monarch transactions to SQLite database."
    )
    parser.add_argument(
        "--csv-path",
        help="Path to Monarch CSV file (defaults to latest download).",
        default=None,
    )
    parser.add_argument(
        "--start-date",
        help="Earliest transaction date to import (YYYY-MM-DD).",
        default=None,
    )
    parser.add_argument(
        "--end-date",
        help="Latest transaction date to import (YYYY-MM-DD).",
        default=None,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Perform a dry run without writing to the database.",
    )
    args = parser.parse_args()

    if args.dry_run:
        print("Dry-run mode enabled. No changes will be made to the database.")
    else:
        print("Changes will be made to the database.")

    main(
        csv_path=args.csv_path,
        dry_run=args.dry_run,
        start_date=args.start_date,
        end_date=args.end_date,
    )
