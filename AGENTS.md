# OpenCode Guidelines for Thesis Paper (Root Project)

## Context & Project Overview
- **Documentation**: The `README.md` file in this directory contains highly relevant and detailed information about the project. Please refer to it for deeper context.
- **Domain**: Hybrid Recommendation Systems combining Collaborative Filtering, Content-Based, and Large Language Models (LLMs).
- **Goal**: Research, build, train, and evaluate recommendation models using MovieLens and TMDB datasets, exposed via a Chatbot API.
- **Environment**: Conda environment named `thesis` (Python 3.11). See `environment.yml` for full dependencies (PyTorch, FastAPI, Ollama, Langchain, Motor, ChromaDB).

## Project Structure & Navigation
This is a monorepo containing several interconnected pieces. Always navigate to the correct directory before making changes:

- **`chat-bot-api/`**: The FastAPI backend that serves the recommendations and interacts with the user/frontend via natural language.
- **`lib/`**: The **Shared Core Library**. Contains shared logic across the entire project including:
  - `database/`: Connections and general repository logic.
  - `model/`: Shared entity models.
  - `recommender/`: Core recommendation algorithms and training logic.
  - `service/` & `util/`: Shared services and utilities.
- **`dags/`**: Apache Airflow DAGs for orchestrating model training (e.g., `cf_emb_update_dag.py`, `thesis_recommenders_upgrade_dag.py`). 
  - ⚠️ **CRITICAL RULE**: The files in `dags/` are the **source of truth**. Do NOT modify DAGs in `~/airflow`. Always modify them here in `thesis-paper/dags/`.
- **`rec-sys-client-lib/`**: Client submodule/library for interacting with or evaluating the recommendation system.
- **`notebooks/`**: Jupyter Notebooks for data preprocessing, exploratory data analysis (EDA), and evaluating model metrics.
- **ML Artifacts**: 
  - `datasets/`: Raw and processed data (e.g., Movielens, TMDB).
  - `weights/`: Trained model weights (PyTorch state dicts, etc).
  - `metrics/`: Evaluation outputs.
  - *Note: Avoid committing large binary files from these directories to Git.*

## Architecture & Data Flow
1. **Data Ingestion & EDA**: Handled mostly via scripts and `notebooks/`.
2. **Model Training (Airflow)**: DAGs orchestrated by Airflow pull new interactions, train Collaborative Filtering models, and generate new embeddings and predictions.
3. **Inference (ChatBot API)**: The user asks for a recommendation. The API uses Ollama (Llama2/3) to generate a textual list, uses ChromaDB (RAG) to match items via Sentence Embeddings, and ranks them using Collaborative Filtering predictions stored in MongoDB.


## Airflow DAGs
The files in the `dags/` directory orchestrate model training and data synchronization. 
- **`1_bert_item_distance_matrix_dag.py`** to **`4_bert_item_distance_matrix_dag.py`**: Compute item distance matrices using various BERT models (`all-MiniLM-L12-v2`, `all-mpnet-base-v2`, etc.).
- **`cf_emb_update_dag.py`**: Generates and updates user and item embeddings. It trains a collaborative filtering model combining pre-train datasets with `chat-bot-api` interactions, then upserts the embeddings into the API's ChromaDB for personalized recommendations.
- **`deep_fm_upgrade_dag.py`**, **`gmf_upgrade_dag.py`**, **`nn_fm_upgrade_dag.py`**, **`thesis_recommenders_upgrade_dag.py`**: These DAGs fetch user interactions from the system, filter for users with >20 interactions, train their respective models (DeepFM, GMF, Neural FM), compute rating matrices, build similarity matrices, and upsert them back to the API.

## Global Conventions
- **Shared Code**: If a feature is used by both the API and the Airflow DAGs, it MUST be placed in the `lib/` directory to avoid code duplication.
- **Dependencies**: Any new dependency should be added to `environment.yml`.

## Code Quality, Architecture & Testing Standards

### SOLID Principles
- **Single Responsibility Principle (SRP)**: Each class must have a single reason to change (e.g., separating HTTP clients, caching decorators, and repositories).
- **Open/Closed Principle (OCP)**: Extend behavior through decorators or subclassing (e.g., using `CachedOllamaApiClient` to add caching to `OllamaApiClient` without modifying it).
- **Dependency Inversion Principle (DIP)**: High-level modules must depend on abstractions/interfaces rather than concrete implementations (e.g., `ChatBotClient` takes an injectable `client` interface). Use dependency injection container pattern (`app_context.py`).

### Design Patterns (Patrones de Diseño)
- **Decorator & Proxy Patterns**: Enforce clean extensibility by decorating clients (e.g. `CachedOllamaApiClient` wrapping `OllamaApiClient`) to inject cross-cutting concerns (caching, logging) without modifying base logic.
- **Factory Pattern**: Use dedicated Factories (e.g., `MovieRecommenderChatBotFactory`) to cleanly abstract complex object instantiation and configuration.
- **Dependency Injection (DI)**: Register and resolve all singletons and services through a central composition root/container (`AppContext`), decoupling object lifetime management from business logic.

### Test-Driven Development (TDD)
- Write unit tests alongside or before implementing new features.
- Design code to be testable: isolate side effects, avoid global state, mock external service dependencies (like MongoDB, ChromaDB, or Ollama APIs).
- Run the test suite via pytest before committing.

### Mutation Testing
- Apply mutation testing principles to evaluate test suite quality.
- Your test assertions must be strong enough to detect changes (mutations) in the business logic code. Do not just assert that a method runs; assert specific side effects, schemas, and returned data.

### Python Coding Best Practices
- **Type Hinting**: Use Pydantic models for data parsing and validation, and enforce Python type annotations (`typing.Dict`, `typing.List`, etc.) for all function signatures.
- **PEP 8 Formatting**: Adhere to clean code standards, readable indentation, snake_case naming for variables and functions, and CamelCase for classes.
- **Shared Decoupling**: Keep the `lib/` package fully detached from specific API framework dependencies (FastAPI, Motor/MongoDB specific connections).
