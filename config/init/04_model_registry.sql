CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE SCHEMA IF NOT EXISTS prudencia;


-- =========================================================
-- MODÈLES ENTRAÎNÉS
-- =========================================================

CREATE TABLE IF NOT EXISTS prudencia.trained_models (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    name VARCHAR(150) NOT NULL,
    version VARCHAR(50) NOT NULL,

    task_type VARCHAR(100) NOT NULL DEFAULT 'ai_act_classification',

    base_model VARCHAR(255) NOT NULL,
    model_path TEXT NOT NULL,

    status VARCHAR(30) NOT NULL DEFAULT 'available',
    is_active BOOLEAN NOT NULL DEFAULT FALSE,

    label_mapping JSONB,

    accuracy DOUBLE PRECISION,
    precision_score DOUBLE PRECISION,
    recall_score DOUBLE PRECISION,
    f1_score DOUBLE PRECISION,
    validation_loss DOUBLE PRECISION,

    training_duration_seconds INTEGER,
    model_size_bytes BIGINT,

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    activated_at TIMESTAMPTZ,

    CONSTRAINT trained_models_name_version_unique
        UNIQUE (name, version),

    CONSTRAINT trained_models_status_check
        CHECK (
            status IN (
                'training',
                'available',
                'failed',
                'archived'
            )
        )
);


-- Un seul modèle actif par tâche
CREATE UNIQUE INDEX IF NOT EXISTS ux_trained_models_active_task
ON prudencia.trained_models (task_type)
WHERE is_active = TRUE;


-- =========================================================
-- HISTORIQUE DES ENTRAÎNEMENTS
-- =========================================================

CREATE TABLE IF NOT EXISTS prudencia.model_training_runs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    model_id UUID REFERENCES prudencia.trained_models(id)
        ON DELETE SET NULL,

    run_name VARCHAR(150) NOT NULL,

    task_type VARCHAR(100) NOT NULL DEFAULT 'ai_act_classification',

    base_model VARCHAR(255) NOT NULL,

    dataset_name VARCHAR(255) NOT NULL,
    dataset_version VARCHAR(50),
    dataset_checksum_sha256 VARCHAR(64),

    text_column VARCHAR(100) NOT NULL,
    label_column VARCHAR(100) NOT NULL,

    total_examples INTEGER NOT NULL,
    training_examples INTEGER,
    validation_examples INTEGER,
    number_of_classes INTEGER NOT NULL,

    classes JSONB,

    epochs INTEGER NOT NULL,
    batch_size INTEGER NOT NULL,
    learning_rate DOUBLE PRECISION NOT NULL,
    validation_split DOUBLE PRECISION NOT NULL,

    device VARCHAR(50),
    status VARCHAR(30) NOT NULL DEFAULT 'pending',

    accuracy DOUBLE PRECISION,
    precision_score DOUBLE PRECISION,
    recall_score DOUBLE PRECISION,
    f1_score DOUBLE PRECISION,

    training_loss DOUBLE PRECISION,
    validation_loss DOUBLE PRECISION,

    training_duration_seconds INTEGER,

    model_output_path TEXT,

    error_message TEXT,
    logs JSONB NOT NULL DEFAULT '[]'::jsonb,

    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT model_training_runs_status_check
        CHECK (
            status IN (
                'pending',
                'running',
                'completed',
                'failed',
                'cancelled'
            )
        ),

    CONSTRAINT model_training_runs_validation_split_check
        CHECK (
            validation_split > 0
            AND validation_split < 1
        ),

    CONSTRAINT model_training_runs_examples_check
        CHECK (
            total_examples > 0
            AND number_of_classes >= 2
        )
);


-- =========================================================
-- INDEX
-- =========================================================

CREATE INDEX IF NOT EXISTS idx_training_runs_created_at
ON prudencia.model_training_runs (created_at DESC);

CREATE INDEX IF NOT EXISTS idx_training_runs_base_model
ON prudencia.model_training_runs (base_model);

CREATE INDEX IF NOT EXISTS idx_training_runs_status
ON prudencia.model_training_runs (status);

CREATE INDEX IF NOT EXISTS idx_trained_models_base_model
ON prudencia.trained_models (base_model);

CREATE INDEX IF NOT EXISTS idx_trained_models_created_at
ON prudencia.trained_models (created_at DESC);


-- =========================================================
-- COMMENTAIRES
-- =========================================================

COMMENT ON TABLE prudencia.model_training_runs IS
'Historique complet des entraînements et benchmarks des modèles IA.';

COMMENT ON TABLE prudencia.trained_models IS
'Registre des modèles entraînés, versionnés et disponibles dans PRUDENCIA.';

COMMENT ON COLUMN prudencia.trained_models.is_active IS
'Indique le modèle actuellement utilisé pour une tâche donnée.';

COMMENT ON COLUMN prudencia.model_training_runs.logs IS
'Journal JSON des différentes étapes de l entraînement.';