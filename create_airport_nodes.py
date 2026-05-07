"""
Script to create Airport nodes in Neo4j from airport_corrdinateds.csv.

Reads airport data with columns: AIRPRT_CD, CNTRY_CD, Latitude_Decimal, Longitude_Decimal, country_cluster_id
Creates nodes with label 'Airport' and properties: airport_code, country_code, lat, long
Applies a unique constraint on airport_code.
"""

import json
import os
import pandas as pd
from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable, TransientError
import time


def load_neo4j_credentials(secrets_path: str) -> dict:
    """Load Neo4j credentials from secrets.json using crew_neo4j key."""
    with open(secrets_path, "r") as f:
        secrets = json.load(f)
    return secrets["crew_neo4j"]


def create_unique_constraint(driver):
    """Create unique constraint on Airport.airport_code."""
    with driver.session() as session:
        session.run("""
            CREATE CONSTRAINT airport_code_unique IF NOT EXISTS
            FOR (a:Airport) REQUIRE a.airport_code IS UNIQUE
        """)
        print("Unique constraint on Airport.airport_code created successfully.")


def read_airport_csv(csv_path: str) -> pd.DataFrame:
    """
    Read airport_corrdinateds.csv and return cleaned DataFrame.
    
    Columns: AIRPRT_CD, CNTRY_CD, Latitude_Decimal, Longitude_Decimal, country_cluster_id
    """
    df = pd.read_csv(
        csv_path,
        dtype={
            'AIRPRT_CD': 'str',
            'CNTRY_CD': 'str',
            'country_cluster_id': 'str'
        }
    )

    # Convert lat/long to decimal floats
    df['Latitude_Decimal'] = pd.to_numeric(df['Latitude_Decimal'], errors='coerce')
    df['Longitude_Decimal'] = pd.to_numeric(df['Longitude_Decimal'], errors='coerce')

    # Drop rows with missing critical data
    df = df.dropna(subset=['AIRPRT_CD', 'Latitude_Decimal', 'Longitude_Decimal'])

    # Strip whitespace from airport codes
    df['AIRPRT_CD'] = df['AIRPRT_CD'].str.strip()
    df['CNTRY_CD'] = df['CNTRY_CD'].fillna('').str.strip()

    print(f"Loaded {len(df):,} airports from CSV.")
    return df


def create_airport_nodes(driver, df: pd.DataFrame, batch_size: int = 1000):
    """
    Create Airport nodes in Neo4j in batches.
    
    Node label: Airport
    Properties: airport_code, country_code, lat, long
    """
    airports_data = []
    for _, row in df.iterrows():
        airports_data.append({
            'airport_code': str(row['AIRPRT_CD']),
            'country_code': str(row['CNTRY_CD']),
            'lat': float(row['Latitude_Decimal']),
            'long': float(row['Longitude_Decimal'])
        })

    total = len(airports_data)
    print(f"Creating {total:,} Airport nodes in Neo4j...")

    for i in range(0, total, batch_size):
        batch = airports_data[i:i + batch_size]
        max_retries = 3

        for attempt in range(max_retries):
            try:
                with driver.session() as session:
                    session.run("""
                        UNWIND $airports AS airport
                        MERGE (a:Airport {airport_code: airport.airport_code})
                        ON CREATE SET
                            a.country_code = airport.country_code,
                            a.lat = airport.lat,
                            a.long = airport.long
                        ON MATCH SET
                            a.country_code = airport.country_code,
                            a.lat = airport.lat,
                            a.long = airport.long
                    """, airports=batch)

                print(f"  Created {min(i + batch_size, total):,} / {total:,} airports")
                break
            except (ServiceUnavailable, TransientError) as e:
                print(f"  Batch {i // batch_size + 1} attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(3)
                else:
                    raise

    print(f"All {total:,} Airport nodes created successfully.")


def verify_import(driver):
    """Verify the airport node count in Neo4j."""
    with driver.session() as session:
        result = session.run("MATCH (a:Airport) RETURN count(a) as count")
        count = result.single()['count']
        print(f"\nVerification: {count:,} Airport nodes in Neo4j.")


if __name__ == "__main__":
    # Load credentials from secrets.json
    secrets_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "secrets.json")
    creds = load_neo4j_credentials(secrets_path)

    NEO4J_URI = creds["connection_string"]
    NEO4J_USERNAME = creds["user"]
    NEO4J_PASSWORD = creds["password"]

    # Path to airport CSV
    CSV_FILE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "airport_coordinates.csv")

    # Create Neo4j driver
    driver = GraphDatabase.driver(
        NEO4J_URI,
        auth=(NEO4J_USERNAME, NEO4J_PASSWORD),
        max_connection_lifetime=10 * 60,
        max_connection_pool_size=10,
        connection_acquisition_timeout=30
    )

    try:
        # Test connection
        with driver.session() as session:
            result = session.run("RETURN 1 as test")
            result.single()['test']
            print("Neo4j connection successful.")

        # Step 1: Create unique constraint
        create_unique_constraint(driver)

        # Step 2: Read CSV
        df = read_airport_csv(CSV_FILE_PATH)

        # Step 3: Create Airport nodes
        create_airport_nodes(driver, df)

        # Step 4: Verify
        verify_import(driver)

    except Exception as e:
        print(f"Error: {e}")
    finally:
        driver.close()
        print("Neo4j connection closed.")
