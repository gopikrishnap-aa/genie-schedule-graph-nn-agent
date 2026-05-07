"""
Join Gemini schedule data (left) with Crew Sequence data (right)
to bring DUTY_PRD_END_FLIGHT_IND from crew seq into the gemini data.

Join keys:
  Left.FLIGHT = Right.FLIGHT_NBR
  Left.DEP_LCL = Right.SCHD_DEP_LCL_TMS
  Left.DEP_STATION = Right.DEP_AIRPRT_IATA_CD
"""

import pandas as pd
import time
import os

# File paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GEMINI_FILE = os.path.join(BASE_DIR, "Gemini Data.txt")
CREW_SEQ_FILE = os.path.join(BASE_DIR, "CrewSeqData_2025_10_11_12.txt")
OUTPUT_FILE = os.path.join(BASE_DIR, "gemini_with_duty_flag.csv")


def load_gemini_data(file_path: str) -> pd.DataFrame:
    """Load Gemini schedule data (left table)."""
    print(f"Loading Gemini data from: {file_path}")
    start = time.time()

    df = pd.read_csv(
        file_path,
        sep='\t',
        dtype=str,
        na_values=['', 'NULL', 'null']
    )

    # Strip whitespace from column names
    df.columns = df.columns.str.strip()

    print(f"  Loaded {len(df):,} rows, {len(df.columns)} columns in {time.time() - start:.1f}s")
    print(f"  Columns: {list(df.columns)}")
    return df


def load_crew_seq_data(file_path: str) -> pd.DataFrame:
    """
    Load Crew Sequence data (right table).
    Only load the columns needed for join + the target column.
    """
    print(f"Loading Crew Seq data from: {file_path}")
    start = time.time()

    df = pd.read_csv(
        file_path,
        sep='\t',
        dtype=str,
        na_values=['', 'NULL', 'null']
    )

    # Strip whitespace from column names
    df.columns = df.columns.str.strip()

    # Keep only the columns needed for join and the target column
    required_cols = ['FLIGHT_NBR', 'SCHD_DEP_LCL_TMS', 'DEP_AIRPRT_IATA_CD', 'DUTY_PRD_END_FLIGHT_IND']
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        print(f"  WARNING: Missing columns in crew seq data: {missing_cols}")
        print(f"  Available columns: {list(df.columns)}")
        raise ValueError(f"Missing required columns: {missing_cols}")

    df = df[required_cols].copy()

    # Deduplicate to avoid many-to-many explosion
    df = df.drop_duplicates(subset=['FLIGHT_NBR', 'SCHD_DEP_LCL_TMS', 'DEP_AIRPRT_IATA_CD'])

    print(f"  Loaded {len(df):,} unique rows in {time.time() - start:.1f}s")
    return df


def join_tables(left_df: pd.DataFrame, right_df: pd.DataFrame) -> pd.DataFrame:
    """
    Left join gemini data with crew seq data.
    
    Join keys:
      Left.FLIGHT = Right.FLIGHT_NBR
      Left.DEP_LCL = Right.SCHD_DEP_LCL_TMS
      Left.DEP_STATION = Right.DEP_AIRPRT_IATA_CD
    """
    print("Joining tables...")
    start = time.time()

    # Strip whitespace from join key values
    left_df['FLIGHT'] = left_df['FLIGHT'].str.strip()
    left_df['DEP_LCL'] = left_df['DEP_LCL'].str.strip()
    left_df['DEP_STATION'] = left_df['DEP_STATION'].str.strip()

    right_df['FLIGHT_NBR'] = right_df['FLIGHT_NBR'].str.strip()
    right_df['SCHD_DEP_LCL_TMS'] = right_df['SCHD_DEP_LCL_TMS'].str.strip()
    right_df['DEP_AIRPRT_IATA_CD'] = right_df['DEP_AIRPRT_IATA_CD'].str.strip()

    # Perform left join
    merged_df = left_df.merge(
        right_df,
        how='left',
        left_on=['FLIGHT', 'DEP_LCL', 'DEP_STATION'],
        right_on=['FLIGHT_NBR', 'SCHD_DEP_LCL_TMS', 'DEP_AIRPRT_IATA_CD']
    )

    # Drop the redundant join columns from the right table
    merged_df = merged_df.drop(columns=['FLIGHT_NBR', 'SCHD_DEP_LCL_TMS', 'DEP_AIRPRT_IATA_CD'], errors='ignore')

    matched = merged_df['DUTY_PRD_END_FLIGHT_IND'].notna().sum()
    total = len(merged_df)
    print(f"  Join complete in {time.time() - start:.1f}s")
    print(f"  Result: {total:,} rows")
    print(f"  Matched: {matched:,} ({matched/total*100:.1f}%)")
    print(f"  Unmatched: {total - matched:,} ({(total-matched)/total*100:.1f}%)")

    return merged_df


if __name__ == "__main__":
    overall_start = time.time()

    # Step 1: Load left table (Gemini)
    gemini_df = load_gemini_data(GEMINI_FILE)

    # Step 2: Load right table (Crew Seq)
    crew_df = load_crew_seq_data(CREW_SEQ_FILE)

    # Step 3: Join
    result_df = join_tables(gemini_df, crew_df)

    # Step 4: Save output
    print(f"\nSaving result to: {OUTPUT_FILE}")
    result_df.to_csv(OUTPUT_FILE, index=False)
    print(f"  Saved {len(result_df):,} rows")

    print(f"\nTotal time: {time.time() - overall_start:.1f}s")
