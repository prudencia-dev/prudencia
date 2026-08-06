CREATE TABLE IF NOT EXISTS prudencia.documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    original_filename TEXT NOT NULL,
    stored_filename TEXT NOT NULL,
    file_path TEXT NOT NULL,
    document_type VARCHAR(100) NOT NULL DEFAULT 'pdf',
    mime_type VARCHAR(255),
    file_size_bytes BIGINT NOT NULL CHECK (file_size_bytes >= 0),
    checksum_sha256 VARCHAR(64) NOT NULL,
    extraction_status VARCHAR(30) NOT NULL DEFAULT 'pending',
    text_extracted BOOLEAN NOT NULL DEFAULT FALSE,
    page_count INTEGER CHECK (page_count IS NULL OR page_count >= 0),
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_documents_created_at
ON prudencia.documents (created_at DESC);

CREATE INDEX IF NOT EXISTS idx_documents_checksum
ON prudencia.documents (checksum_sha256);

CREATE TABLE IF NOT EXISTS prudencia.document_chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID NOT NULL REFERENCES prudencia.documents(id)
        ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL CHECK (chunk_index >= 0),
    content TEXT NOT NULL,
    word_count INTEGER NOT NULL DEFAULT 0 CHECK (word_count >= 0),
    token_count INTEGER CHECK (token_count IS NULL OR token_count >= 0),
    chroma_collection TEXT,
    chroma_document_id TEXT,
    embedding_model TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT document_chunks_document_index_unique
        UNIQUE (document_id, chunk_index)
);

CREATE INDEX IF NOT EXISTS idx_document_chunks_document_id
ON prudencia.document_chunks (document_id);

CREATE INDEX IF NOT EXISTS idx_document_chunks_chroma_document_id
ON prudencia.document_chunks (chroma_document_id);
