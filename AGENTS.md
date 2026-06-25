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

## Extra instructions
### Agent development cycle (default, unless overridden)
#### Mandatory Workflow Enforcement (Tool Usage)
- **You MUST use the `todowrite` tool** at the very beginning of every new feature, bugfix, or refactor request.
- You are strictly forbidden from writing production code before populating your TODO list with the following exact 5-step lifecycle:
  1. **Analyze & Plan:** Analyze the requirement and propose a plan to the user. STOP and wait for approval.
  2. **TDD (RED):** Write failing unit/integration tests that specify the desired behavior.
  3. **TDD (GREEN):** Implement the minimum production code necessary to pass the tests.
  4. **Refactor & Quality Gate:** Refactor the code applying SOLID principles, Clean Code, and explicitly handling errors/exceptions.
  5. **Final Verification:** Run all quality checks (`pytest`, linters).
- Keep the `todowrite` status updated in real-time as you progress through each step. Never skip the RED phase.
- **Plan once:** Before coding, propose a TODO list oriented to iterative implementation and STOP. Ask for approval once. After approval, proceed without asking for the plan again unless new information invalidates it.
- **Test-first (when supported):** If the repo has an existing test framework, write/update tests that specify the new behavior before implementing the production change.
- **Quality gate (must answer "yes" before finishing):**
    - **Task alignment:** Does the change meet every requirement from the original request (and nothing unrelated)?
    - **Tests for new logic:** Did I add/adjust unit tests covering the success path and relevant error or edge cases?
    - **Idiomatic + consistent:** Does the implementation follow Python and repo conventions?
    - **Clarity + simplicity:** Is the code easy to read and minimizes complexity?
    - **Error handling:** Are failure modes handled explicitly, with no silent failures?
- **Final verification:** Run the applicable validation commands (e.g., `pytest`).

### Architecture instructions

#### General Software Design & OOP Principles

**1. The SOLID Principles**
* **Single Responsibility (SRP):** A class should have one, and only one, reason to change. Keep components strictly focused.
* **Open/Closed (OCP):** Software entities should be open for extension but closed for modification.
* **Liskov Substitution (LSP):** Subclasses must be substitutable for their base classes without breaking the application.
* **Interface Segregation (ISP):** Prefer multiple, highly specific interfaces over a single, general-purpose one.
* **Dependency Inversion (DIP):** High-level modules should not depend on low-level modules; both should depend on abstractions. Depend on interfaces, not on concrete implementations.

**2. DRY (Don't Repeat Yourself)**
* Avoid duplicating code or logic. Every piece of knowledge must have a single, unambiguous representation.

**3. KISS (Keep It Simple, Stupid)**
* Systems work best when they are kept simple rather than made complicated.

**4. YAGNI (You Aren't Gonna Need It)**
* Always implement things when you actually need them, never when you just foresee that you might need them in the future.

**5. Law of Demeter (Principle of Least Knowledge)**
* An object should assume as little as possible about the structure of other objects. It should only interact with its immediate dependencies.

**6. High Cohesion**
* Ensure that all the methods and properties within a class are strongly related and focused on a unified purpose.

**7. Low Coupling**
* Minimize the dependencies between different classes. Changes in one class should have little to no impact on other classes.

#### Python Object-Oriented Programming (OOP) Best Practices

**1. Type Hinting Strictness**
* Always use type hints (`typing` module, or built-in generic types) for function signatures, arguments, and class attributes.

**2. Leverage Dataclasses and Pydantic**
* Use `@dataclass` or Pydantic `BaseModel` for classes whose primary purpose is to hold data. This significantly reduces boilerplate and provides built-in validation.

**3. Strict Encapsulation**
* Hide implementation details. Use single leading underscores `_` to indicate internal/private methods and instance variables. Expose only what is necessary through public methods or properties.

**4. Avoid Mutable Default Arguments**
* Never use mutable structures (like `[]` or `{}`) as default arguments in functions to prevent unexpected side effects across function calls.

**5. Composition over Inheritance**
* Avoid inheriting from concrete classes whenever possible. If you need to extend behavior, prefer composition and dependency injection.

**6. Explicit Exception Handling**
* Catch specific exceptions rather than broad `Exception` classes. Fail fast and explicitly. Never use bare `except:` clauses.

### Design Patterns (Patrones de Diseño)
- **Decorator & Proxy Patterns**: Enforce clean extensibility by decorating clients (e.g. `CachedOllamaApiClient` wrapping `OllamaApiClient`) to inject cross-cutting concerns (caching, logging) without modifying base logic.
- **Factory Pattern**: Use dedicated Factories (e.g., `MovieRecommenderChatBotFactory`) to cleanly abstract complex object instantiation and configuration.
- **Dependency Injection (DI)**: Register and resolve all singletons and services through a central composition root/container (`AppContext`), decoupling object lifetime management from business logic.

#### Development methodology and code quality

**1. Test-Driven Development (TDD)**
- **Mandatory Approach:** All system changes (new features, bug fixes, refactoring) must be designed and implemented using **Test-Driven Development (TDD)**.
- **Red-Green-Refactor Cycle:**
  1. **Red:** Write a unit or integration test that fails, specifying the desired behavior.
  2. **Green:** Implement the minimum code necessary to make the test pass successfully.
  3. **Refactor:** Clean and optimize the implemented code, ensuring it follows the style guidelines and application architecture without breaking the tests.

**2. Mutation Testing**
- **Test Suite Validation:** To ensure the quality and actual effectiveness of the tests, apply **Mutation Testing** principles.
- **Goal:** Validate that the test suite is robust and fails in the presence of mutations. If tests still pass after changing logic, they are not rigorously validating the expected behavior.
