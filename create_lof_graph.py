"""
Create LOF flight sequence graphs in Neo4j from gemini_with_duty_flag.csv.

Filters: SCHEDULE = 'Proposal 2025-10 F0'
Groups by: LOF, EQP
Orders by: PER_SEQ

Graph structure per LOF:
  (LOF) -[HAS_SEQUENCE]-> (Flight1) -[NEXT]-> (Flight2) -[NEXT]-> ... -[ENDS_AT]-> (LOF)
  Each Flight connects to existing Airport nodes:
    (Flight) -[DEPARTS_FROM]-> (Airport {airport_code})
    (Flight) -[ARRIVES_AT]-> (Airport {airport_code})

Connection: crew_neo4j from secrets.json
"""

import json
import os
import time
import pandas as pd
from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable, TransientError


# --------------- Configuration ---------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SECRETS_PATH = os.path.join(BASE_DIR, "secrets.json")
CSV_FILE = os.path.join(BASE_DIR, "gemini_with_duty_flag.csv")
SCHEDULE_FILTER = "Proposal 2025-10 F0"


def load_credentials() -> dict:
    """Load Neo4j credentials from secrets.json using crew_neo4j key."""
    with open(SECRETS_PATH, "r") as f:
        secrets = json.load(f)
    return secrets["crew_neo4j"]


def create_driver(creds: dict):
    """Create Neo4j driver."""
    return GraphDatabase.driver(
        creds["connection_string"],
        auth=(creds["user"], creds["password"]),
        max_connection_lifetime=5 * 60,
        max_connection_pool_size=5,
        connection_acquisition_timeout=60
    )


def load_and_filter_data(csv_path: str, schedule_filter: str) -> pd.DataFrame:
    """Load CSV, filter by SCHEDULE, group by LOF/EQP, order by PER_SEQ."""
    print(f"Loading data from: {csv_path}")
    df = pd.read_csv(csv_path, dtype=str, na_values=['', 'NULL', 'null', 'nan'])
    df.columns = df.columns.str.strip()

    print(f"  Total rows: {len(df):,}")

    # Filter by SCHEDULE
    df['SCHEDULE'] = df['SCHEDULE'].str.strip()
    df = df[df['SCHEDULE'] == schedule_filter].copy()
    print(f"  After filtering SCHEDULE='{schedule_filter}': {len(df):,} rows")

    # Filter by ML_RG = 'ML'
    df['ML_RG'] = df['ML_RG'].str.strip()
    df = df[df['ML_RG'] == 'ML'].copy()
    print(f"  After filtering ML_RG='ML': {len(df):,} rows")

    # Convert PER_SEQ to numeric for proper ordering
    df['PER_SEQ_NUM'] = pd.to_numeric(df['PER_SEQ'], errors='coerce').fillna(0).astype(int)

    # Sort by LOF, EQP, PER_SEQ
    df = df.sort_values(by=['LOF', 'EQP', 'PER_SEQ_NUM']).reset_index(drop=True)

    unique_lofs = df.groupby(['LOF', 'EQP']).ngroups
    print(f"  Unique LOF+EQP groups: {unique_lofs:,}")

    return df


def create_constraints_and_indexes(driver):
    """Create constraints and indexes for LOF and Flight nodes."""
    with driver.session() as session:
        # Unique constraint on Flight: dep_lcl + lof + eqp + per_seq
        session.run("""
            CREATE CONSTRAINT flight_unique IF NOT EXISTS
            FOR (f:Flight) REQUIRE (f.dep_lcl, f.lof, f.eqp, f.per_seq) IS UNIQUE
        """)
        # Index on Flight for sequence lookups
        session.run("""
            CREATE INDEX flight_lof_seqidx IF NOT EXISTS
            FOR (f:Flight) ON (f.lof, f.sequence_index)
        """)
        print("Constraints and indexes created.")


def process_single_lof(driver, lof_number: str, eqp: str, flights: list):
    """
    Process a single LOF group: create nodes and relationships.
    Each LOF processed in its own session to avoid transaction size issues.
    """
    with driver.session() as session:
        # Step 1: Create LOF node
        session.run("""
            MERGE (l:LOF {lof_number: $lof_number, eqp: $eqp})
        """, lof_number=lof_number, eqp=eqp)

        # Step 2: Create all Flight nodes with MERGE on unique keys (dep_lcl, lof, eqp, per_seq)
        flights_with_idx = [{**f, 'seq_idx': idx} for idx, f in enumerate(flights)]
        session.run("""
            UNWIND $flights AS flight
            MERGE (f:Flight {dep_lcl: flight.dep_lcl, lof: flight.lof, eqp: flight.eqp, per_seq: flight.per_seq})
            ON CREATE SET
                f.dep_station = flight.dep_station,
                f.arr_station = flight.arr_station,
                f.arr_lcl = flight.arr_lcl,
                f.duty_prd_end_flight_ind = flight.duty_prd_end_flight_ind,
                f.skd_blk = flight.skd_blk,
                f.fleet = flight.fleet,
                f.body_type = flight.body_type,
                f.flight_number = flight.flight_number,
                f.sequence_index = flight.seq_idx,
                f.mogt = flight.mogt,
                f.turn_time = flight.turn_time,
                f.time_over_mogt = flight.time_over_mogt
            ON MATCH SET
                f.dep_station = flight.dep_station,
                f.arr_station = flight.arr_station,
                f.arr_lcl = flight.arr_lcl,
                f.duty_prd_end_flight_ind = flight.duty_prd_end_flight_ind,
                f.skd_blk = flight.skd_blk,
                f.fleet = flight.fleet,
                f.body_type = flight.body_type,
                f.flight_number = flight.flight_number,
                f.sequence_index = flight.seq_idx,
                f.mogt = flight.mogt,
                f.turn_time = flight.turn_time,
                f.time_over_mogt = flight.time_over_mogt
        """, flights=flights_with_idx)

        # Step 3: Link LOF -> first Flight
        session.run("""
            MATCH (l:LOF {lof_number: $lof_number, eqp: $eqp})
            MATCH (f:Flight {lof: $lof_number, sequence_index: 0})
            CREATE (l)-[:FIRST_LEG]->(f)
        """, lof_number=lof_number, eqp=eqp)

        # Step 4: Link last Flight -> LOF
        last_idx = len(flights) - 1
        session.run("""
            MATCH (l:LOF {lof_number: $lof_number, eqp: $eqp})
            MATCH (f:Flight {lof: $lof_number, sequence_index: $last_idx})
            CREATE (f)-[:LAST_LEG]->(l)
        """, lof_number=lof_number, eqp=eqp, last_idx=last_idx)

        # Step 5: Chain flights with NEXT relationships
        if len(flights) > 1:
            pairs = [{'from_idx': i, 'to_idx': i + 1} for i in range(len(flights) - 1)]
            session.run("""
                UNWIND $pairs AS pair
                MATCH (a:Flight {lof: $lof_number, sequence_index: pair.from_idx})
                MATCH (b:Flight {lof: $lof_number, sequence_index: pair.to_idx})
                CREATE (a)-[:NEXT]->(b)
            """, lof_number=lof_number, pairs=pairs)

        # Step 6: Connect to Airport nodes (DEPARTS_FROM)
        session.run("""
            MATCH (f:Flight {lof: $lof_number})
            WHERE f.dep_station <> ''
            WITH f
            MATCH (a:Airport {airport_code: f.dep_station})
            MERGE (f)-[:DEPARTS_FROM]->(a)
        """, lof_number=lof_number)

        # Step 7: Connect to Airport nodes (ARRIVES_AT)
        session.run("""
            MATCH (f:Flight {lof: $lof_number})
            WHERE f.arr_station <> ''
            WITH f
            MATCH (a:Airport {airport_code: f.arr_station})
            MERGE (f)-[:ARRIVES_AT]->(a)
        """, lof_number=lof_number)


def create_lof_graphs(creds: dict, df: pd.DataFrame):
    """Create all LOF sequence graphs, one LOF at a time with fresh connections."""
    grouped = df.groupby(['LOF', 'EQP'])
    total_groups = len(grouped)
    print(f"\nCreating {total_groups:,} LOF sequence graphs...")

    start_time = time.time()
    processed = 0
    failed = 0
    driver = create_driver(creds)

    for (lof_number, eqp), group_df in grouped:
        # Prepare flight data for this LOF
        flights = []
        for _, row in group_df.iterrows():
            flights.append({
                'dep_station': str(row.get('DEP_STATION', '') or ''),
                'arr_station': str(row.get('ARR_STATION', '') or ''),
                'duty_prd_end_flight_ind': str(row.get('DUTY_PRD_END_FLIGHT_IND', '') or ''),
                'skd_blk': str(row.get('SKD_BLK', '') or ''),
                'fleet': str(row.get('FLEET', '') or ''),
                'body_type': str(row.get('BODY_TYPE', '') or ''),
                'eqp': str(row.get('EQP', '') or ''),
                'dep_lcl': str(row.get('DEP_LCL', '') or ''),
                'arr_lcl': str(row.get('ARR_LCL', '') or ''),
                'per_seq': int(row.get('PER_SEQ_NUM', 0)),
                'lof': str(lof_number),
                'flight_number': str(row.get('FLIGHT', '') or ''),
                'mogt': int(pd.to_numeric(row.get('MOGT', 0), errors='coerce') or 0),
                'turn_time': int(pd.to_numeric(row.get('TURN_TIME', 0), errors='coerce') or 0),
                'time_over_mogt': int(pd.to_numeric(row.get('TIME_OVER_MOGT', 0), errors='coerce') or 0),
            })

        # Process this LOF with retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
                process_single_lof(driver, str(lof_number), str(eqp), flights)
                processed += 1
                break
            except (ServiceUnavailable, TransientError) as e:
                if attempt < max_retries - 1:
                    time.sleep(2)
                    # Reconnect driver on transient failures
                    try:
                        driver.close()
                    except Exception:
                        pass
                    driver = create_driver(creds)
                else:
                    failed += 1
                    print(f"  FAILED LOF {lof_number}: {e}")
            except Exception as e:
                failed += 1
                print(f"  ERROR LOF {lof_number}: {e}")
                break

        # Reconnect every 100 LOFs to keep connection fresh
        if processed % 100 == 0 and processed > 0:
            elapsed = time.time() - start_time
            rate = processed / elapsed if elapsed > 0 else 0
            print(f"  Progress: {processed:,}/{total_groups:,} | {rate:.1f} LOFs/sec | Elapsed: {elapsed:.0f}s")
            try:
                driver.close()
            except Exception:
                pass
            driver = create_driver(creds)

    try:
        driver.close()
    except Exception:
        pass

    elapsed = time.time() - start_time
    print(f"\nCompleted: {processed:,} LOF graphs created, {failed:,} failed | Total: {elapsed:.1f}s")


def verify_import(driver):
    """Verify the import results."""
    with driver.session() as session:
        lof_count = session.run("MATCH (l:LOF) RETURN count(l) as count").single()['count']
        flight_count = session.run("MATCH (f:Flight) RETURN count(f) as count").single()['count']
        next_count = session.run("MATCH ()-[r:NEXT]->() RETURN count(r) as count").single()['count']
        dep_count = session.run("MATCH ()-[r:DEPARTS_FROM]->() RETURN count(r) as count").single()['count']
        arr_count = session.run("MATCH ()-[r:ARRIVES_AT]->() RETURN count(r) as count").single()['count']

        print(f"\n--- Import Verification ---")
        print(f"  LOF nodes: {lof_count:,}")
        print(f"  Flight nodes: {flight_count:,}")
        print(f"  NEXT relationships: {next_count:,}")
        print(f"  DEPARTS_FROM relationships: {dep_count:,}")
        print(f"  ARRIVES_AT relationships: {arr_count:,}")


if __name__ == "__main__":
    print("=" * 60)
    print("LOF Flight Sequence Graph Creator")
    print("=" * 60)

    # Load credentials
    creds = load_credentials()
    print(f"Neo4j URI: {creds['connection_string']}")

    # Create driver
    driver = create_driver(creds)

    try:
        # Test connection
        with driver.session() as session:
            session.run("RETURN 1 as test").single()
            print("Neo4j connection successful.\n")

        # Load and filter data
        df = load_and_filter_data(CSV_FILE, SCHEDULE_FILTER)

        if df.empty:
            print("No data found after filtering. Exiting.")
            exit(0)

        # Create constraints
        create_constraints_and_indexes(driver)

        # Create LOF graphs (one at a time with reconnects)
        create_lof_graphs(creds, df)

        # Verify
        verify_import(driver)

    except Exception as e:
        print(f"Error: {e}")
        raise
    finally:
        driver.close()
        print("\nNeo4j connection closed.")
