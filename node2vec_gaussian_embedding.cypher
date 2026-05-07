-- =============================================================
-- Node2Vec Embedding with Gaussian-Weighted Turn Time
-- =============================================================
-- Purpose: Use TURN_TIME (already in minutes on Flight nodes) to compute
-- Gaussian weights on NEXT relationships, then run Node2Vec to generate
-- flight sequence embeddings that capture overnight/RON patterns.
--
-- Data from CSV:
--   TURN_TIME: actual ground time between flights (minutes)
--   MOGT: Minimum Operationally required Ground Time (minutes)
--   TIME_OVER_MOGT: TURN_TIME - MOGT (excess ground time in minutes)
--
-- Gaussian weight: w = exp(-(turn_time - mu)^2 / (2 * sigma^2))
--   mu    = 240 minutes (4 hours) — ideal layover
--   sigma = 120 minutes (2 hours) — spread
--
-- Interpretation:
--   - Turn time ~4hrs → weight ~1.0 (ideal connection, high walk probability)
--   - Turn time ~1hr  → weight ~0.1 (tight connection, walks less likely)
--   - Turn time >10hr → weight ~0.0 (likely overnight — walks avoid these)
--
-- Node2Vec parameters:
--   p = 1.0 (return parameter): controls revisiting departure base
--   q = 0.5 (in-out parameter): biased toward exploration (multi-hop overnights)
--
-- REQUIRES: GDS plugin
-- Run sequentially in Neo4j Browser / Aura
-- =============================================================


-- =============================================
-- STEP 1: Set turn_time / MOGT on NEXT relationships from Flight nodes
-- =============================================
-- The NEXT relationship connects Flight A → Flight B.
-- Flight B's turn_time is the ground time BEFORE that flight departs
-- (i.e., the gap between A's arrival and B's departure).

MATCH (a:Flight)-[r:NEXT]->(b:Flight)
WHERE b.turn_time IS NOT NULL AND b.turn_time > 0
SET r.turnTime = b.turn_time,
    r.mogt = b.mogt,
    r.timeOverMogt = b.time_over_mogt;


-- =============================================
-- STEP 2: Apply Gaussian weight to NEXT relationships
-- =============================================
-- Gaussian: w = exp(-(turnTime - mu)^2 / (2 * sigma^2))
-- mu = 240 min (4 hours), sigma = 120 min (2 hours)
--
-- This means:
--   turnTime = 240min (4hr) → weight = 1.0 (ideal, walks prefer these)
--   turnTime =  60min (1hr) → weight = 0.32 (tight, walks less likely)
--   turnTime = 480min (8hr) → weight = 0.32 (long, potential overnight)
--   turnTime = 720min (12hr)→ weight = 0.04 (very long, almost certainly overnight)

MATCH (a:Flight)-[r:NEXT]->(b:Flight)
WHERE r.turnTime IS NOT NULL
WITH r, toFloat(r.turnTime) AS t,
     240.0 AS mu,
     120.0 AS sigma
WITH r, t, mu, sigma,
     exp(-1.0 * (t - mu) * (t - mu) / (2.0 * sigma * sigma)) AS gaussianWeight
SET r.gaussianWeight = gaussianWeight;


-- For edges where turn_time is missing or 0, set a low default weight
MATCH (a:Flight)-[r:NEXT]->(b:Flight)
WHERE r.gaussianWeight IS NULL
SET r.gaussianWeight = 0.01;


-- =============================================
-- STEP 3: Verify weight distribution
-- =============================================
MATCH ()-[r:NEXT]->()
WHERE r.turnTime IS NOT NULL
RETURN
  round(avg(r.turnTime), 1) AS avgTurnTime,
  round(avg(r.gaussianWeight), 4) AS avgWeight,
  round(min(r.gaussianWeight), 4) AS minWeight,
  round(max(r.gaussianWeight), 4) AS maxWeight,
  count(r) AS totalEdges,
  sum(CASE WHEN r.turnTime > 480 THEN 1 ELSE 0 END) AS potentialOvernights,
  sum(CASE WHEN r.turnTime > 840 THEN 1 ELSE 0 END) AS potentialDoubleOvernights,
  sum(CASE WHEN r.timeOverMogt > 300 THEN 1 ELSE 0 END) AS excessiveGroundTime;


-- =============================================
-- STEP 4: Project weighted graph for Node2Vec
-- =============================================
CALL gds.graph.drop('flightNode2Vec', false);

CALL gds.graph.project(
  'flightNode2Vec',
  'Flight',
  {
    NEXT: {
      type: 'NEXT',
      properties: {
        weight: {
          property: 'gaussianWeight',
          defaultValue: 0.01
        }
      }
    }
  },
  { memory: '8GB' }
);


-- =============================================
-- STEP 5: Run Node2Vec to generate embeddings
-- =============================================
-- Parameters:
--   embeddingDimension: 64 (compact but expressive)
--   walkLength: 10 (captures multi-hop sequences within an LOF)
--   walksPerNode: 20 (sufficient coverage per flight)
--   returnFactor (p): 1.0 — neutral about returning to prior node
--   inOutFactor (q): 0.5 — biased toward exploration (finds multi-hop overnights)
--   relationshipWeightProperty: 'weight' — uses Gaussian turn-time weights
--
-- Low q means the walk prefers to explore NEW nodes → captures extended
-- sequences where crew flies multiple hops away from base.
-- High-weight edges (normal ~4hr turns) are traversed more often →
-- flights with normal connections cluster together in embedding space.
-- Low-weight edges (long overnights) are rarely traversed →
-- flights across overnight boundaries land in DIFFERENT embedding clusters.

CALL gds.node2vec.write('flightNode2Vec', {
  embeddingDimension: 64,
  walkLength: 10,
  walksPerNode: 20,
  returnFactor: 1.0,
  inOutFactor: 0.5,
  relationshipWeightProperty: 'weight',
  writeProperty: 'node2vecEmbedding'
});


-- =============================================
-- STEP 6: Verify embeddings
-- =============================================
MATCH (f:Flight)
WHERE f.node2vecEmbedding IS NOT NULL
RETURN count(f) AS flightsWithEmbeddings,
       size(f.node2vecEmbedding) AS embeddingDimension
LIMIT 1;

-- Sample embeddings with turn time context
MATCH (a:Flight)-[r:NEXT]->(b:Flight)
WHERE b.node2vecEmbedding IS NOT NULL
RETURN a.flight_number AS fromFlight,
       b.flight_number AS toFlight,
       a.dep_station AS fromDep,
       b.arr_station AS toArr,
       r.turnTime AS turnTimeMin,
       r.mogt AS mogtMin,
       r.timeOverMogt AS excessMin,
       r.gaussianWeight AS weight,
       b.isDutyEndFlight AS isDutyEnd,
       b.node2vecEmbedding[0..5] AS embeddingSample
LIMIT 10;


-- =============================================
-- STEP 7: Correlation check — turn time vs duty end
-- =============================================
-- Expect: flights following long turn times are more likely duty-end
MATCH (a:Flight)-[r:NEXT]->(b:Flight)
WHERE r.turnTime IS NOT NULL
RETURN b.isDutyEndFlight AS isDutyEnd,
       round(avg(r.turnTime), 1) AS avgTurnTime,
       round(avg(r.mogt), 1) AS avgMOGT,
       round(avg(r.timeOverMogt), 1) AS avgExcessOverMOGT,
       round(avg(r.gaussianWeight), 4) AS avgGaussWeight,
       count(*) AS cnt
ORDER BY isDutyEnd;


-- =============================================
-- STEP 8: CLEANUP (run when done, keeps embeddings on nodes)
-- =============================================
-- CALL gds.graph.drop('flightNode2Vec', false);
