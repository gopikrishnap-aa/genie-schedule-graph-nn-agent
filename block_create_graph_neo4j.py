import pandas as pd
from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable, TransientError
import gc
import time
from typing import Iterator

class OptimizedFlightImporter:
    def __init__(self, uri, username, password):
        # Store connection details instead of creating persistent driver
        self.uri = uri
        self.username = username
        self.password = password
        
    def create_driver(self):
        """Create a new Neo4j driver on demand"""
        return GraphDatabase.driver(
            self.uri, 
            auth=(self.username, self.password),
            max_connection_lifetime=10 * 60,  # 5 minutes
            max_connection_pool_size=10,
            connection_acquisition_timeout=30
        )
    def test_connection(self):
        """Test Neo4j connection before starting import"""
        driver = None
        try:
            driver = self.create_driver()
            with driver.session() as session:
                result = session.run("RETURN 1 as test")
                test_value = result.single()['test']
                print(f"Neo4j connection successful (test value: {test_value})")
                return True
        except Exception as e:
            print(f"Neo4j connection failed: {e}")
            return False
        finally:
            if driver:
                driver.close()
    
    def close(self):
        """Close method - no persistent connection to close"""
        pass
    
    def create_constraints(self):
        """Create constraints and indexes for optimal performance with on-demand connection"""
        driver = None
        max_retries = 3
        for attempt in range(max_retries):
            try:
                driver = self.create_driver()
                with driver.session() as session:
                    # Unique constraint on airport code
                    session.run("""
                        CREATE CONSTRAINT airport_code_unique IF NOT EXISTS 
                        FOR (a:Airports) REQUIRE a.code IS UNIQUE
                    """)
                    
                    # Indexes for better query performance
                    session.run("""
                        CREATE INDEX flight_departure_date_idx IF NOT EXISTS 
                        FOR ()-[f:FLIGHT]-() ON (f.departure_date_time)
                    """)
                    
                    session.run("""
                        CREATE INDEX flight_carrier_idx IF NOT EXISTS 
                        FOR ()-[f:FLIGHT]-() ON (f.carrier)
                    """)
                    
                    print(" Constraints and indexes created")
                    return
            except (ServiceUnavailable, TransientError) as e:
                print(f" Constraint creation attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    print(f" Retrying in 5 seconds...")
                    time.sleep(5)
                else:
                    raise
            finally:
                if driver:
                    driver.close()
                    driver = None
    
    def read_csv_efficiently(self, csv_file_path: str, chunk_size: int = 10000) -> Iterator[pd.DataFrame]:
        """
        Read CSV in chunks with only required columns for memory efficiency
        """
       
        required_columns = [
            'departure_station',     # Airport.code (origin)
            'arrival_station',       # Airport.code (destination)
            'orig_lat',           # Airport.lat (origin)
            'orig_long',          # Airport.long (origin)
            'dest_lat',      # Airport.lat (destination)
            'dest_long',     # Airport.long (destination)
            'carrier',              # Edge.carrier
            'great_dist_flown',     # Edge.great_circle_distance
            'fleet_code',           # Edge.fleet_code
            'sub_fleet_code',       # Edge.sub_fleet_code
            'bearing',              # Edge.bearing
            'scheduled_departure_local_timestamp',  # Edge.departure_date_time
            'scheduled_arrival_local_timestamp',    # Edge.arrival_date_time
            'actual_air'            # Edge.actual_air
        ]
        
        print(f"Reading CSV in chunks of {chunk_size:,} rows with only {len(required_columns)} required columns...")
        
        try:
            # Read only required columns in chunks
            chunk_reader = pd.read_csv(
                csv_file_path,
                usecols=required_columns,
                chunksize=chunk_size,
                dtype={
                    'departure_station': 'str',
                    'arrival_station': 'str',
                    'carrier': 'str',
                    'fleet_code': 'str',
                    'sub_fleet_code': 'str',
                    'orig_lat': 'float32',      # Use float32 instead of float64 for memory
                    'orig_long': 'float32',
                    'dest_lat': 'float32',
                    'dest_long': 'float32',
                    'great_dist_flown': 'float32',
                    'bearing': 'float32',
                    'actual_air': 'float32'         # Changed from int32 to float32 to handle NaN
                },
                na_values=['', 'NULL', 'null', 'NaN', 'nan']  # Explicitly handle various NA representations
            )
            
            total_processed = 0
            for chunk_num, chunk in enumerate(chunk_reader, 1):
                # Clean the chunk - remove rows with missing critical data
                chunk_cleaned = chunk.dropna(subset=['departure_station', 'arrival_station'])
                
                total_processed += len(chunk_cleaned)
                print(f"Processing chunk {chunk_num}: {len(chunk_cleaned):,} valid rows (Total: {total_processed:,})")
                
                yield chunk_cleaned
                
                # Force garbage collection every 10 chunks
                if chunk_num % 10 == 0:
                    gc.collect()
                    
        except Exception as e:
            print(f"Error reading CSV: {e}")
            raise
    
    def extract_unique_airports_efficiently(self, csv_file_path: str, chunk_size: int = 15000) -> dict:
        """
        Extract unique airports efficiently without loading entire dataset
        """
        print("Extracting unique airports...")
        airports_dict = {}
        
        for chunk in self.read_csv_efficiently(csv_file_path, chunk_size):
            # Process origin airports
            origin_airports = chunk[['departure_station', 'orig_lat', 'orig_long']].dropna()
            for _, row in origin_airports.iterrows():
                code = row['departure_station']
                if code not in airports_dict:
                    airports_dict[code] = {
                        'code': code,
                        'lat': float(row['orig_lat']),
                        'long': float(row['orig_long']),
                        'country_code': ''
                    }
            
            # Process destination airports
            dest_airports = chunk[['arrival_station', 'dest_lat', 'dest_long']].dropna()
            for _, row in dest_airports.iterrows():
                code = row['arrival_station']
                if code not in airports_dict:
                    airports_dict[code] = {
                        'code': code,
                        'lat': float(row['dest_lat']),
                        'long': float(row['dest_long']),
                        'country_code': ''
                    }
        
        print(f"Found {len(airports_dict):,} unique airports")
        return airports_dict
    
    def create_airports_batch(self, airports_dict: dict, batch_size: int = 1000):
        """Create airport nodes with on-demand connections"""
        print("Creating airport nodes...")
        
        airports_list = list(airports_dict.values())
        
        for i in range(0, len(airports_list), batch_size):
            batch = airports_list[i:i+batch_size]
            driver = None
            
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    driver = self.create_driver()
                    with driver.session() as session:
                        session.run("""
                            UNWIND $airports AS airport
                            MERGE (a:Airports {code: airport.code})
                            ON CREATE SET 
                                a.lat = airport.lat,
                                a.long = airport.long,
                                a.country_code = airport.country_code
                        """, airports=batch)
                    
                    print(f"Created {min(i+batch_size, len(airports_list)):,} / {len(airports_list):,} airports")
                    break
                except (ServiceUnavailable, TransientError) as e:
                    print(f"Airport batch {i//batch_size + 1} attempt {attempt + 1} failed: {e}")
                    if attempt < max_retries - 1:
                        print(f"Retrying batch in 3 seconds...")
                        time.sleep(3)
                    else:
                        print(f"Failed to create airport batch after {max_retries} attempts")
                        raise
                finally:
                    if driver:
                        driver.close()
                        driver = None
    
    def create_flights_efficiently(self, csv_file_path: str, chunk_size: int = 5000, neo4j_batch_size: int = 1000):
        """Create flight relationships with on-demand connections"""
        print("Creating flight relationships...")
        
        total_flights = 0
        start_time = time.time()
        batch_count = 0
        
        for chunk in self.read_csv_efficiently(csv_file_path, chunk_size):
            # Process chunk in smaller Neo4j batches
            for i in range(0, len(chunk), neo4j_batch_size):
                batch = chunk.iloc[i:i+neo4j_batch_size]
                batch_count += 1
                
                # Prepare flight data
                flights_data = []
                for _, row in batch.iterrows():
                    if (pd.notna(row['departure_station']) and 
                        pd.notna(row['arrival_station']) and
                        row['departure_station'] != row['arrival_station'] and
                        pd.notna(row['scheduled_departure_local_timestamp']) and
                        pd.notna(row['scheduled_arrival_local_timestamp'])):
                        
                        # Clean datetime strings for Neo4j compatibility
                        dep_time = str(row['scheduled_departure_local_timestamp']).replace('.000Z', 'Z')
                        arr_time = str(row['scheduled_arrival_local_timestamp']).replace('.000Z', 'Z')
                        
                        # Skip if datetime cleaning resulted in invalid values
                        if dep_time == 'nan' or arr_time == 'nan' or dep_time == 'None' or arr_time == 'None':
                            continue
                            
                        flights_data.append({
                            'origin': str(row['departure_station']),
                            'destination': str(row['arrival_station']),
                            'carrier': str(row.get('carrier', '')),
                            'great_circle_distance': int(row['great_dist_flown']) if pd.notna(row['great_dist_flown']) else 0,
                            'fleet_code': str(row.get('fleet_code', '')),
                            'sub_fleet_code': str(row.get('sub_fleet_code', '')),
                            'bearing': float(row['bearing']) if pd.notna(row['bearing']) else 0.0,
                            'departure_date_time': dep_time,
                            'arrival_date_time': arr_time,
                            'actual_air': int(row['actual_air']) if pd.notna(row['actual_air']) else 0
                        })
                
                # Create relationships with on-demand connection and retry logic
                if flights_data:
                    driver = None
                    max_retries = 3
                    for attempt in range(max_retries):
                        try:
                            driver = self.create_driver()
                            with driver.session() as session:
                                session.run("""
                                    UNWIND $flights AS flight
                                    MATCH (origin:Airports {code: flight.origin})
                                    MATCH (dest:Airports {code: flight.destination})
                                    CREATE (origin)-[:FLIGHT {
                                        carrier: flight.carrier,
                                        great_circle_distance: flight.great_circle_distance,
                                        fleet_code: flight.fleet_code,
                                        sub_fleet_code: flight.sub_fleet_code,
                                        bearing: flight.bearing,
                                        departure_date_time: datetime(flight.departure_date_time),
                                        arrival_date_time: datetime(flight.arrival_date_time),
                                        actual_air: flight.actual_air,
                                        flight_date: date(datetime(flight.departure_date_time))
                                    }]->(dest)
                                """, flights=flights_data)
                            
                            total_flights += len(flights_data)
                            break  # Success, exit retry loop
                        
                        except (ServiceUnavailable, TransientError) as e:
                            print(f"Flight batch {batch_count} attempt {attempt + 1} failed: {e}")
                            if attempt < max_retries - 1:
                                print(f"Retrying batch in 3 seconds...")
                                time.sleep(3)
                            else:
                                print(f"Failed to create flight batch after {max_retries} attempts")
                                raise
                        finally:
                            if driver:
                                driver.close()
                                driver = None
                
                # Progress update every 10,000 flights
                if total_flights % 10000 == 0 and total_flights > 0:
                    elapsed = time.time() - start_time
                    rate = total_flights / elapsed
                    print(f"Created {total_flights:,} flights | Rate: {rate:.1f} flights/sec | Elapsed: {elapsed:.1f}s | Batch: {batch_count}")
        
        print(f"Total flights created: {total_flights:,}")
        return total_flights
    
    def import_optimized(self, csv_file_path: str, csv_chunk_size: int = 8000, neo4j_batch_size: int = 800):
        """
        Main optimized import function
        """
        print("Starting optimized 7M row import...")
        start_time = time.time()
        
        try:
            # Step 1: Create constraints
            self.create_constraints()
            
            # Step 2: Extract unique airports efficiently
            airports_dict = self.extract_unique_airports_efficiently(csv_file_path, csv_chunk_size)
            
            # Step 3: Create airport nodes
            self.create_airports_batch(airports_dict)
            
            # Step 4: Create flight relationships
            total_flights = self.create_flights_efficiently(csv_file_path, csv_chunk_size, neo4j_batch_size)
            
            # Final verification
            self.verify_import()
            
            elapsed = time.time() - start_time
            print(f"\n Import completed successfully!")
            print(f" Total time: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
            print(f"Average rate: {total_flights/elapsed:.1f} flights/second")
            
        except Exception as e:
            print(f" Error during optimized import: {e}")
            raise
    
    def verify_import(self):
        """Quick verification of imported data with on-demand connection"""
        driver = None
        try:
            driver = self.create_driver()
            with driver.session() as session:
                # Count nodes and relationships
                airport_result = session.run("MATCH (a:Airports) RETURN count(a) as count")
                airport_count = airport_result.single()['count']
                
                flight_result = session.run("MATCH ()-[f:FLIGHT]->() RETURN count(f) as count")
                flight_count = flight_result.single()['count']
                
                print(f"\n Import Summary:")
                print(f"Airports: {airport_count:,}")
                print(f"Flights: {flight_count:,}")
                if airport_count > 0:
                    print(f" Average flights per airport: {flight_count/airport_count:.1f}")
        finally:
            if driver:
                driver.close()



        
if __name__ == "__main__":
    # Load credentials from secrets.json
    import json
    import os

    secrets_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "secrets.json")
    with open(secrets_path, "r") as f:
        secrets = json.load(f)

    crew_neo4j = secrets["crew_neo4j"]
    NEO4J_URI = crew_neo4j["connection_string"]
    NEO4J_USERNAME = crew_neo4j["user"]
    NEO4J_PASSWORD = crew_neo4j["password"]
    CSV_FILE_PATH = r"C:\ML Code\ML_RG_IATA_Validation\featured_data.csv" 
    
    CSV_CHUNK_SIZE = 800000  
    NEO4J_BATCH_SIZE = 10000
    
    importer = OptimizedFlightImporter(NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD)
    
    try:
        # Test connection first
        if not importer.test_connection():
            print("Cannot proceed - fix connection issues first")
            exit(1)
        
        # Run optimized import
        importer.import_optimized(
            csv_file_path=CSV_FILE_PATH,
            csv_chunk_size=CSV_CHUNK_SIZE,
            neo4j_batch_size=NEO4J_BATCH_SIZE
        )
        
    except Exception as e:
        print(f"Import failed: {e}")
    finally:
        importer.close()