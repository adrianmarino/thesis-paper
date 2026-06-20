# OpenCode Guidelines for ChatBot API (Thesis Project)

## Context & Project Overview
- **Documentation**: The `README.md` file in the project root directory (`../README.md`) contains highly relevant information about the thesis background, models, and theory. Refer to it when needed.
- **Domain**: Hybrid Recommendation Systems combining Large Language Models (LLMs), Collaborative Filtering (CF), and Content-Based filtering.
- **Shared Code**: The `../lib` directory contains shared Python packages (`database`, `model`, `recommender`, `service`) used across the entire thesis project.
- **Airflow Interaction**: The API receives updated embeddings and predictions periodically from an external Airflow DAG training process. 

## Inference Flow (How it works)
1. **User Profile**: Solves the cold-start problem. Profiles are managed via MongoDB.
2. **LLM Generation**: Uses Ollama API (Llama2/Llama3) to generate raw movie recommendations based on user prompts.
3. **RAG & Matching**: The LLM's text output is matched against real items in ChromaDB using Sentence Embeddings (`all-mpnet-base-v2` or `mxbai-embed-large`). 
4. **Ordering**: Results are ranked using Collaborative Filtering predictions stored in MongoDB.

## API Contract / Endpoints
Base URL: `/api/v1`

### OpenAPI / Swagger Documentation
Because this project uses FastAPI, it automatically generates and serves its OpenAPI specification. When the server is running (usually on port `8080`), you can access the live contract here:
- **Swagger UI (Interactive)**: `http://<host>:8080/docs`
- **ReDoc**: `http://<host>:8080/redoc`
- **OpenAPI JSON**: `http://<host>:8080/openapi.json`


## Airflow DAGs (External)
The Airflow DAGs that feed data and train models for this API are located in the parent project's `../dags/` directory. The most relevant ones for the ChatBot API are:
- **`cf_emb_update_dag.py`**: Generates and updates the embeddings representing users and items. It uses the tasks defined in this project (`dag_task/cf_emb_update_task.py`) to train a Collaborative Filtering model with new API interactions and upserts the fresh embeddings into ChromaDB.
- **Recommender Upgrade DAGs** (e.g., `thesis_recommenders_upgrade_dag.py`): Fetch user interactions, train deep learning recommenders (DeepFM, GMF, etc.), and push similarity matrices back to the database.

## Tech Stack
- **Framework**: FastAPI + Uvicorn + Pydantic.
- **Databases**: 
  - **MongoDB** (Motor): User profiles, chat histories, interactions, and CF predictions.
  - **ChromaDB**: Item content embeddings and collaborative filtering (CF) embeddings.
- **Environment**: Conda (`thesis` environment).

## Architecture & Code Conventions
- **AppContext (`app_context.py`)**: Dependency Injection container. **Rule:** Every new Service, Repository, or Job must be initialized and registered here.
- **Handlers (`handlers/`)**: FastAPI endpoints (API Contract). Do NOT put business logic here. Extract parameters and delegate to `ctx.<service_name>`.
- **Services (`services/`)**: Core business logic (e.g. `interaction_service.py`, `recommendation_chat_service.py`).
- **Repositories (`repository/`)**: Persistence layer wrappers (`MongoRepository`, `ChromaRepository`).
- **Models & Mappers**: `models/` contains Pydantic schemas. `mappers/` contains translation logic between DB entities and Models.

## Scripts & Operations
- Configurations are defined in `config.conf`.
- **Start Service**: `bin/start` (Activates conda environment and runs Uvicorn in the background).
- **Stop Service**: `bin/stop` (Kills Uvicorn process).

## Code Quality, Architecture & Testing Standards

### SOLID Principles in the API
- **Single Responsibility (SRP)**: Keep FastAPI handlers thin (parsing inputs and delegating directly to services). Do not write business logic or DB calls inside handler endpoints.
- **Dependency Inversion (DIP)**: All dependencies of services (repositories, external clients) must be injected via the constructor. Do not instantiate dependencies directly inside services. Use `app_context.py` as the composition root.
- **Open/Closed (OCP)**: Decorate clients or services (e.g., adding caching decorators like `CachedOllamaApiClient` around the raw client) to extend functionality without modifying existing code.

### Design Patterns (Patrones de Diseño)
- **Proxy/Decorator Pattern**: Used to apply caching around the Ollama client seamlessly (`CachedOllamaApiClient` wraps the standard `OllamaApiClient` without changing its contract).
- **Factory Pattern**: Utilized to instantiate recommender models and chatbot clients (e.g. `MovieRecommenderChatBotFactory`), centralizing construction logic.
- **Dependency Injection (DI)**: Applied systematically through the constructor of all handlers, services, and repositories to invert dependencies (DIP), orchestrated via `AppContext`.

### Test-Driven Development (TDD)
- Implement test cases alongside endpoints and services under `tests/`.
- Mock external systems (Ollama API, MongoDB, ChromaDB) in unit tests using unittest.mock or pytest-mock to ensure tests run fast and deterministically without external infrastructure requirements.

### Mutation Testing
- Test suite reliability should be verified by ensuring that altering code logic (e.g. changing comparison operators, removing lines) causes tests to fail (Mutation Testing). 
- Avoid loose assertions (e.g., asserting `True` or simply checking if a function was called without checking its arguments and results).

### Python & FastAPI Best Practices
- Enforce full type annotations on all endpoints, services, and helper methods.
- Use Pydantic's `BaseModel` for request validation and response models to leverage FastAPI's auto-generated Swagger/OpenAPI documentation.
- Leverage async/await concurrency properly. Ensure any blocking synchronous calls (like synchronous client queries) are run in separate threads using executors to avoid blocking the Uvicorn event loop.
