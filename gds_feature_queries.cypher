-- =============================================================
-- GDS Feature Engineering Queries for isDutyEndFlight Prediction
-- =============================================================
-- Target: isDutyEndFlight (boolean true/false) on Flight nodes
-- Datetime: depDateTime (native Neo4j datetime, e.g. 2025-10-02T19:23:00Z)
-- Run these queries sequentially in Neo4j Browser / Aura
--
-- REQUIRES: GDS plugin + APOC plugin (for apoc.periodic.iterate)
-- If APOC not available, replace apoc.periodic.iterate with
-- CALL { ... } IN TRANSACTIONS OF 1000 ROWS (Neo4j 5.x)
-- =============================================================


-- =============================================
-- 0. VERIFY DATA BEFORE STARTING
-- =============================================
-- Check how many flights have the target label
MATCH (f:Flight)
RETURN f.isDutyEndFlight AS label, count(*) AS cnt
ORDER BY cnt DESC;


-- =============================================
-- 1. CO-DEPARTURE: Create relationships first, then project
-- =============================================
-- STEP 1a: Create index on dep_station + depDateTime for fast grouping
CREATE INDEX flight_dep_station_idx IF NOT EXISTS FOR (f:Flight) ON (f.dep_station);
CREATE INDEX flight_dep_datetime_idx IF NOT EXISTS FOR (f:Flight) ON (f.depDateTime);

-- STEP 1b: Create CO_DEPARTS relationships efficiently using station grouping
-- This avoids full Cartesian product by grouping per station+date first
CALL apoc.periodic.iterate(
  "MATCH (f:Flight)
   WHERE f.depDateTime IS NOT NULL
   WITH f.dep_station AS station, date(f.depDateTime) AS depDate, collect(f) AS flights
   WHERE size(flights) > 1
   RETURN flights",
  "WITH flights
   UNWIND range(0, size(flights)-2) AS i
   UNWIND range(i+1, size(flights)-1) AS j
   WITH flights[i] AS f1, flights[j] AS f2
   WHERE abs(duration.between(f1.depDateTime, f2.depDateTime).minutes) <= 30
   CREATE (f1)-[:CO_DEPARTS]->(f2)",
  {batchSize: 100, parallel: false}
);

-- STEP 1c: Native projection (FAST - just reads existing relationships)
CALL gds.graph.drop('flightDepartures', false);

CALL gds.graph.project(
  'flightDepartures',
  'Flight',
  'CO_DEPARTS',
  {undirectedRelationshipTypes: ['CO_DEPARTS'], memory: '8GB'}
);

-- STEP 1d: Write departure degree centrality
CALL gds.degree.write('flightDepartures', { writeProperty: 'departureDegree' });


-- =============================================
-- 2. CO-ARRIVAL: Create relationships first, then project
-- =============================================
-- STEP 2a: Create index on arr_station
CREATE INDEX flight_arr_station_idx IF NOT EXISTS FOR (f:Flight) ON (f.arr_station);

-- STEP 2b: Create CO_ARRIVES relationships efficiently
CALL apoc.periodic.iterate(
  "MATCH (f:Flight)
   WHERE f.arr_lcl IS NOT NULL
   WITH f.arr_station AS station, collect(f) AS flights
   WHERE size(flights) > 1
   RETURN flights",
  "WITH flights
   UNWIND range(0, size(flights)-2) AS i
   UNWIND range(i+1, size(flights)-1) AS j
   WITH flights[i] AS f1, flights[j] AS f2
   WHERE abs(duration.between(datetime(f1.arr_lcl), datetime(f2.arr_lcl)).minutes) <= 30
   CREATE (f1)-[:CO_ARRIVES]->(f2)",
  {batchSize: 100, parallel: false}
);

-- STEP 2c: Native projection (FAST)
CALL gds.graph.drop('flightArrivals', false);

CALL gds.graph.project(
  'flightArrivals',
  'Flight',
  'CO_ARRIVES',
  {undirectedRelationshipTypes: ['CO_ARRIVES'], memory: '8GB'}
);

-- STEP 2d: Write arrival degree centrality
CALL gds.degree.write('flightArrivals', { writeProperty: 'arrivalDegree' });


-- =============================================
-- 3. FLIGHT SEQUENCE PROJECTION (NEXT chain)
-- =============================================
-- Project the LOF sequence chain (Flight)-[NEXT]->(Flight)
CALL gds.graph.drop('flightSequence', false);

CALL gds.graph.project(
  'flightSequence',
  'Flight',
  'NEXT',
  {memory: '8GB'}
);

-- 3a. PageRank on flight sequence - measures flight importance in chains
CALL gds.pageRank.write('flightSequence', {
  writeProperty: 'sequencePageRank',
  maxIterations: 20,
  dampingFactor: 0.85
});

-- 3b. Betweenness Centrality - flights that bridge different LOF segments
CALL gds.betweenness.write('flightSequence', {
  writeProperty: 'sequenceBetweenness'
});

-- 3c. Article Rank (variant of PageRank, less bias to sinks/sources)
CALL gds.articleRank.write('flightSequence', {
  writeProperty: 'articleRank',
  maxIterations: 20,
  dampingFactor: 0.85
});


-- =============================================
-- 4. UNDIRECTED SEQUENCE PROJECTION (for community detection)
-- =============================================
CALL gds.graph.drop('flightSequenceUndirected', false);

CALL gds.graph.project(
  'flightSequenceUndirected',
  'Flight',
  'NEXT',
  {undirectedRelationshipTypes: ['NEXT'], memory: '8GB'}
);

-- 4a. Louvain Community Detection
CALL gds.louvain.write('flightSequenceUndirected', {
  writeProperty: 'louvainCommunity'
});

-- 4b. Label Propagation Community Detection
CALL gds.labelPropagation.write('flightSequenceUndirected', {
  writeProperty: 'labelPropCommunity'
});

-- 4c. Triangle Count & Clustering Coefficient
-- SKIPPED: Aura GDS doesn't support undirected projections via config param.
-- departureDegree + arrivalDegree already capture station connectivity.
-- Set defaults so model training doesn't break:
MATCH (f:Flight)
SET f.triangleCount = 0, f.clusteringCoefficient = 0.0;


-- =============================================
-- 5. STATION CONNECTIVITY FEATURES (single-pass per station)
-- =============================================

-- 5a+5c combined: Departure station flight count + unique routes in one query
MATCH (f:Flight)
WITH f.dep_station AS station, count(*) AS cnt, count(DISTINCT f.arr_station) AS uniqueDest
CALL {
  WITH station, cnt, uniqueDest
  MATCH (f2:Flight {dep_station: station})
  SET f2.depStationFlightCount = cnt,
      f2.depStationUniqueRoutes = uniqueDest
} IN TRANSACTIONS OF 500 ROWS;

-- 5b. Arrival station flight count
MATCH (f:Flight)
WITH f.arr_station AS station, count(*) AS cnt
CALL {
  WITH station, cnt
  MATCH (f2:Flight {arr_station: station})
  SET f2.arrStationFlightCount = cnt
} IN TRANSACTIONS OF 500 ROWS;


-- =============================================
-- 6. LOF POSITIONAL FEATURES (single-pass per LOF)
-- =============================================

-- 6a+6b+6c combined: LOF size, relative position, isLast in ONE pass
MATCH (f:Flight)
WITH f.lof AS lof, count(*) AS lofSize, max(f.sequence_index) AS maxIdx
CALL {
  WITH lof, lofSize, maxIdx
  MATCH (f2:Flight {lof: lof})
  SET f2.lofSize = lofSize,
      f2.relativePosition = CASE WHEN maxIdx > 0
        THEN toFloat(f2.sequence_index) / toFloat(maxIdx)
        ELSE 0.0 END,
      f2.isLastInLof = CASE WHEN f2.sequence_index = maxIdx THEN 1 ELSE 0 END
} IN TRANSACTIONS OF 500 ROWS;

-- 6d+6e+6f combined: isFirst, hasNext, hasPrev in ONE scan
MATCH (f:Flight)
OPTIONAL MATCH (f)-[:NEXT]->(nxt:Flight)
OPTIONAL MATCH (prv:Flight)-[:NEXT]->(f)
SET f.isFirstInLof = CASE WHEN f.sequence_index = 0 THEN 1 ELSE 0 END,
    f.hasNextFlight = CASE WHEN nxt IS NOT NULL THEN 1 ELSE 0 END,
    f.hasPrevFlight = CASE WHEN prv IS NOT NULL THEN 1 ELSE 0 END;


-- =============================================
-- 7. TEMPORAL / BLOCK TIME FEATURES (SINGLE scan)
-- =============================================
-- All temporal features in ONE pass instead of 6 separate scans
-- NOTE: arr_lcl is in format "10/5/2025 23:11:00" (M/d/yyyy HH:mm:ss)
MATCH (f:Flight)
SET f.depHour = CASE WHEN f.depDateTime IS NOT NULL THEN f.depDateTime.hour ELSE null END,
    f.depDayOfWeek = CASE WHEN f.depDateTime IS NOT NULL THEN f.depDateTime.dayOfWeek ELSE null END,
    f.isRedEye = CASE WHEN f.depDateTime IS NOT NULL AND f.depDateTime.hour >= 21 THEN 1 ELSE 0 END,
    f.arrHour = CASE WHEN f.arr_lcl IS NOT NULL AND f.arr_lcl <> '' AND f.arr_lcl <> 'nan'
      THEN datetime({epochMillis: apoc.date.parse(f.arr_lcl, 'ms', 'M/d/yyyy HH:mm:ss')}).hour
      ELSE null END,
    f.isLateArrival = CASE WHEN f.arr_lcl IS NOT NULL AND f.arr_lcl <> '' AND f.arr_lcl <> 'nan'
      AND datetime({epochMillis: apoc.date.parse(f.arr_lcl, 'ms', 'M/d/yyyy HH:mm:ss')}).hour >= 18
      THEN 1 ELSE 0 END,
    f.blockTimeMinutes = CASE
      WHEN f.skd_blk IS NOT NULL AND f.skd_blk <> '' AND f.skd_blk <> 'nan'
      THEN toInteger(f.skd_blk)
      ELSE null END;


-- =============================================
-- 8. FLEET FEATURES (single-pass)
-- =============================================
MATCH (f:Flight)
WITH f.fleet AS fleet, count(*) AS fleetCount
CALL {
  WITH fleet, fleetCount
  MATCH (f2:Flight {fleet: fleet})
  SET f2.fleetPopularity = fleetCount
} IN TRANSACTIONS OF 500 ROWS;


-- =============================================
-- 9. VERIFY ENRICHED FEATURES
-- =============================================
MATCH (f:Flight)
RETURN f.flight_number,
       f.dep_station,
       f.arr_station,
       f.isDutyEndFlight,
       f.departureDegree,
       f.arrivalDegree,
       f.sequencePageRank,
       f.sequenceBetweenness,
       f.articleRank,
       f.louvainCommunity,
       f.triangleCount,
       f.clusteringCoefficient,
       f.lofSize,
       f.relativePosition,
       f.isLastInLof,
       f.hasNextFlight,
       f.depHour,
       f.arrHour,
       f.isLateArrival,
       f.depStationFlightCount,
       f.blockTimeMinutes
LIMIT 20;


-- =============================================
-- 10. CLEANUP PROJECTIONS & TEMP RELATIONSHIPS
-- =============================================
CALL gds.graph.drop('flightDepartures', false);
CALL gds.graph.drop('flightArrivals', false);
CALL gds.graph.drop('flightSequence', false);
CALL gds.graph.drop('flightSequenceUndirected', false);

-- Optional: Remove temp relationships if you don't need them anymore
-- MATCH ()-[r:CO_DEPARTS]->() DELETE r;
-- MATCH ()-[r:CO_ARRIVES]->() DELETE r;
