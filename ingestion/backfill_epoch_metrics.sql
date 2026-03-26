-- Backfill train_epoch_metric from existing train_step_metric and checkpoint_sample_metric.
-- Uses train_dataset_len and batch_size from model_parameter to derive epoch boundaries.
-- Creates a temporary table first for inspection before committing.

-- Step 1: Get steps_per_epoch for each (model_id, run_id)
CREATE OR REPLACE TEMP TABLE run_epoch_info AS
SELECT
    ds.model_id,
    ds.run_id,
    ds.value AS dataset_len,
    bs.value AS batch_size,
    CEIL(ds.value::FLOAT / bs.value) AS steps_per_epoch
FROM local.model_parameter_int ds
JOIN local.model_parameter_int bs
    ON ds.model_id = bs.model_id AND ds.run_id = bs.run_id
WHERE ds.name = 'train_dataset_len'
  AND bs.name = 'train_config.batch_size';

-- Diagnostic: show runs that are missing epoch info
SELECT 'runs_with_epoch_info' AS label, COUNT(DISTINCT (model_id, run_id)) AS count FROM run_epoch_info
UNION ALL
SELECT 'runs_missing_epoch_info', COUNT(DISTINCT (model_id, run_id))
FROM local.train_step_metric_float m
WHERE NOT EXISTS (
    SELECT 1 FROM run_epoch_info rei
    WHERE rei.model_id = m.model_id AND rei.run_id = m.run_id
);

-- Step 2: Aggregate training step metrics into epoch-level summaries
CREATE OR REPLACE TEMP TABLE backfill_epoch_metrics AS
SELECT
    m.model_id,
    m.run_id,
    MAX(m.timestamp) AS timestamp,
    (FLOOR(m.step / rei.steps_per_epoch) + 1)::INTEGER AS epoch,
    MAX(m.step)::INTEGER AS step,
    m.name,
    ts.dataset,
    'train' AS dataset_split,
    AVG(m.value)::FLOAT AS mean,
    MIN(m.value)::FLOAT AS min,
    MAX(m.value)::FLOAT AS max,
    COUNT(*)::INTEGER AS count
FROM local.train_step_metric_float m
JOIN run_epoch_info rei
    ON m.model_id = rei.model_id AND m.run_id = rei.run_id
LEFT JOIN (
    SELECT DISTINCT model_id, run_id, dataset
    FROM local.train_steps
) ts ON m.model_id = ts.model_id AND m.run_id = ts.run_id
WHERE NOT EXISTS (
    SELECT 1 FROM local.train_epoch_metric e
    WHERE e.model_id = m.model_id AND e.run_id = m.run_id
)
GROUP BY m.model_id, m.run_id, FLOOR(m.step / rei.steps_per_epoch) + 1, m.name, ts.dataset;

-- Step 3: Aggregate validation metrics into epoch-level summaries
INSERT INTO backfill_epoch_metrics
SELECT
    csm.model_id,
    r.id AS run_id,
    MAX(csm.timestamp) AS timestamp,
    (FLOOR(csm.step / rei.steps_per_epoch) + 1)::INTEGER AS epoch,
    MAX(csm.step)::INTEGER AS step,
    csm.name,
    csm.dataset,
    'val' AS dataset_split,
    AVG(csm.mean)::FLOAT AS mean,
    MIN(csm.mean)::FLOAT AS min,
    MAX(csm.mean)::FLOAT AS max,
    COUNT(*)::INTEGER AS count
FROM local.checkpoint_sample_metric_float csm
JOIN local.runs r ON r.model_id = csm.model_id
JOIN run_epoch_info rei
    ON csm.model_id = rei.model_id AND r.id = rei.run_id
WHERE (csm.name LIKE '%loss%' OR csm.name LIKE '%accuracy%')
AND NOT EXISTS (
    SELECT 1 FROM local.train_epoch_metric e
    WHERE e.model_id = csm.model_id AND e.run_id = r.id
)
GROUP BY csm.model_id, r.id, FLOOR(csm.step / rei.steps_per_epoch) + 1, csm.name, csm.dataset;

-- Step 4: Inspect
SELECT count(*) AS total_rows FROM backfill_epoch_metrics;

SELECT * FROM backfill_epoch_metrics ORDER BY model_id, epoch, name LIMIT 50;

-- Step 5: When satisfied, insert into the real table and log the backfill:

-- CREATE TABLE IF NOT EXISTS local.backfill_log (
--     backfill_id BIGINT,
--     timestamp TIMESTAMPTZ,
--     description TEXT,
--     model_id BIGINT,
--     run_id BIGINT,
--     rows_inserted INTEGER
-- );
--
-- INSERT INTO local.train_epoch_metric BY NAME SELECT * FROM backfill_epoch_metrics;
--
-- INSERT INTO local.backfill_log
-- SELECT
--     hash(now()::TEXT) AS backfill_id,
--     now() AS timestamp,
--     'epoch_metrics_from_step_data' AS description,
--     model_id,
--     run_id,
--     COUNT(*)::INTEGER AS rows_inserted
-- FROM backfill_epoch_metrics
-- GROUP BY model_id, run_id;
