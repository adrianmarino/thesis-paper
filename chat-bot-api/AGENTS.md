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

- **Profiles (`/profiles`)**: 
  - `GET /`, `GET /{email}`, `POST /`, `PUT /{email}`, `DELETE /{email}`
- **Interactions (`/interactions`)**: 
  - `GET /`, `GET /users/{user_id}`, `POST /`, `POST /bulk`, `GET /make/{user_id}/{item_id}/{rating}`
  - `DELETE /users/{user_id}`, `DELETE /users/{user_id}/items/{item_id}`
- **Items (`/items`)**: 
  - `GET /`, `GET /{item_id}`, `POST /`, `POST /bulk`, `PUT /embeddings/content/build`, `DELETE /{item_id}`
- **Recommendations (`/recommendations`)**: 
  - `GET /models`, `POST /`
- **Chat Histories (`/histories`)**: 
  - `GET /{email}`, `DELETE /{email}`
- **Recommenders (`/recommenders`)**: 
  - `PUT /train`


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
