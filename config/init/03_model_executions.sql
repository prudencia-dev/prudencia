CREATE TABLE IF NOT EXISTS prudencia.model_executions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_type VARCHAR(100) NOT NULL,
    model_name VARCHAR(255) NOT NULL,
    model_version VARCHAR(100) NOT NULL,
    task_name VARCHAR(150) NOT NULL,
    execution_type VARCHAR(50) NOT NULL DEFAULT 'training',
    dataset_name TEXT NOT NULL DEFAULT '',
    dataset_rows INTEGER NOT NULL DEFAULT 0 CHECK (dataset_rows >= 0),
    input_data JSONB NOT NULL DEFAULT '{}'::jsonb,
    output_data JSONB NOT NULL DEFAULT '{}'::jsonb,
    execution_time_ms BIGINT NOT NULL DEFAULT 0
        CHECK (execution_time_ms >= 0),
    is_active BOOLEAN NOT NULL DEFAULT FALSE,
    success BOOLEAN NOT NULL DEFAULT FALSE,
    error_message TEXT,
    executed_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_model_executions_executed_at
ON prudencia.model_executions (executed_at DESC);

CREATE INDEX IF NOT EXISTS idx_model_executions_model
ON prudencia.model_executions (model_type, model_name);

CREATE UNIQUE INDEX IF NOT EXISTS ux_model_executions_active_model
ON prudencia.model_executions (model_type, model_name)
WHERE is_active = TRUE;
