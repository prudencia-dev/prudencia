# Schéma visuel de l'architecture du code source

## Dépendances entre couches

![Dépendances entre les couches du code source](images/dependances-couches-code.svg)

```mermaid
flowchart TD
    subgraph Frontend["data/streamlit"]
        APP["app.py\nNavigation"]
        PAGES["pages/\nRAG, modèles, rapport"]
        MODULES["modules/\nbase, fine-tuning, historique"]
        CLIENT["services/api.py\nClient HTTP"]
        APP --> PAGES
        APP --> MODULES
        PAGES --> CLIENT
        MODULES --> CLIENT
    end

    CLIENT -->|"HTTP JSON / multipart"| MAIN

    subgraph Backend["data/fastapi/app"]
        MAIN["main.py\nComposition FastAPI"]
        MID["request_context.py\nlogging_config.py"]
        ROUTERS["api/\nContrats HTTP"]
        SERVICES["services/\nLogique métier"]
        DOMAIN["domain/\nObjets du rapport"]
        REPO["repositories/\nAccès transactionnel"]
        AI["ai/\nDL, embeddings, Chroma"]
        DB["database.py\nConnexion PostgreSQL"]

        MAIN --> MID
        MAIN --> ROUTERS
        ROUTERS --> SERVICES
        SERVICES --> DOMAIN
        SERVICES --> REPO
        SERVICES --> AI
        REPO --> DB
        SERVICES --> DB
    end

    DB --> PG[("PostgreSQL")]
    AI --> CH[("ChromaDB")]
    AI --> MODELS[("Modèles / cache Hugging Face")]
```

## Arborescence fonctionnelle du backend

![Arborescence fonctionnelle du backend](images/arborescence-backend.svg)

```mermaid
flowchart LR
    ROOT["app/"] --> MAIN["main.py"]
    ROOT --> API["api/"]
    ROOT --> SVC["services/"]
    ROOT --> AIM["ai/"]
    ROOT --> DOM["domain/"]
    ROOT --> REP["repositories/"]
    ROOT --> INFRA["Infrastructure transverse"]

    API --> A1["document_analysis.py"]
    API --> A2["fine_tuning.py"]
    API --> A3["model_registry.py"]
    API --> A4["rag.py"]
    API --> A5["reports.py"]
    API --> A6["training_history.py"]
    API --> A7["error_responses.py"]

    SVC --> S1["Document / PDF / Chunking"]
    SVC --> S2["RAG / Upload security"]
    SVC --> S3["Analysis orchestrator / Report builder"]
    SVC --> S4["Model registry / Selection"]
    SVC --> S5["Training history / Indexing progress"]

    AIM --> I1["bge_m3.py"]
    AIM --> I2["camembert.py"]
    AIM --> I3["rag.py"]
    AIM --> I4["embedding_config.py"]
    AIM --> FT["fine_tuning/"]
    FT --> F1["dataset.py"]
    FT --> F2["trainer.py"]
    FT --> F3["experiment.py"]
    FT --> F4["benchmark.py"]

    DOM --> D1["report.py"]
    REP --> R1["base_repository.py"]
    INFRA --> X1["database.py"]
    INFRA --> X2["config.py"]
    INFRA --> X3["request_context.py"]
    INFRA --> X4["logging_config.py"]
```

## Chaîne d'appel par cas d'usage

![Chaînes d'appel principales](images/chaines-appels.svg)

```mermaid
flowchart LR
    subgraph Indexation["Indexer un PDF"]
        IR["api/rag.py"] --> IRS["services/rag_service.py"]
        IRS --> US["services/upload_security.py"]
        IRS --> DS["services/document_service.py"]
        IRS --> PS["services/pdf_service.py"]
        IRS --> CS["services/chunking_service.py"]
        IRS --> BGE["ai/bge_m3.py"]
        IRS --> CR["ai/rag.py"]
    end

    subgraph Rapport["Générer un rapport"]
        RR["api/reports.py"] --> AO["services/analysis_orchestrator.py"]
        AO --> RB["services/report_builder.py"]
        RB --> DR["domain/report.py"]
    end

    subgraph Entrainement["Fine-tuner un modèle"]
        FR["api/fine_tuning.py"] --> DM["ai/fine_tuning/dataset.py"]
        FR --> TR["ai/fine_tuning/trainer.py"]
        TR --> EM["ai/fine_tuning/experiment.py"]
        FR --> TH["services/training_history_service.py"]
    end
```

## Dépendances transverses

![Traitement transversal des requêtes](images/traitement-transversal-requetes.svg)

```mermaid
flowchart TB
    REQ["Toute requête HTTP"] --> CTX["RequestContextMiddleware"]
    CTX --> RID["ContextVar request_id"]
    RID --> ROUTE["Route FastAPI"]
    ROUTE --> ERR["raise_api_error"]
    ROUTE --> LOG["logger prudencia.*"]
    ERR --> LOG
    LOG --> FMT["JsonFormatter / TextFormatter"]
    FMT --> STDOUT["stdout Docker"]
```

## Règles de dépendance recommandées

- Streamlit dépend de l'API HTTP, jamais directement de PostgreSQL ou ChromaDB.
- Les routeurs dépendent des services, mais les services ne doivent pas dépendre des routeurs.
- Les objets du domaine ne doivent pas importer FastAPI, Streamlit ou les clients de base de données.
- Les moteurs `ai/` encapsulent les bibliothèques externes et les modèles.
- La gestion des secrets et chemins reste dans la configuration et les services d'infrastructure.
- Les nouvelles requêtes SQL devraient être regroupées progressivement dans `repositories/`.
