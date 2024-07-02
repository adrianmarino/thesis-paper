
<p align="left">

<b>University of Buenos Aires</br>Faculty of Exact and natural sciences</br>Master in Data Mining and Knowledge Discovery</b>
</p>

# Collaborative and hybrid recommendation systems</h1>


This study aims to compare different approaches to recommendation based on collaborative and hybrid filtering (i.e., a combination of collaborative and content-based filters), explaining the advantages and disadvantages of each approach, as well as their architecture and operation for each proposed model. In the realm of hybrid models or ensembles, experiments were conducted with ensembles of different types including LLM(Large language models), content-based models, and collaborative filtering-based models. The MovieLens and TMDB datasets were chosen as the basis for defining a dataset, as they are classic datasets commonly used for comparing recommendation models.

<p align="center">
  <img src="https://github.com/adrianmarino/thesis-paper/blob/master/images/logo.png?raw=true"  height="800" />
</p>

## Table of Contents

1. [Requisites](#requisites)
2. [Hypothesis](#hypothesis)
3. [Documents](#documents)
4. [Models](#models)
5. [Metrics](#metrics)
6. [Data](#data)
7. [Notebooks](#notebooks)
    1. [Data pre-processing & analysis](#data-pre-processing--analysis)
    2. [Recommendation Models](#recommendation-models)
        1. [Evaluation](#evaluation)
        2. [Baseline](#baseline)
        3. [Collaborative Filtering](#collaborative-filtering)
        4. [Content Based](#content-based)
        5. [Ensembles](#ensembles)
    3. [Extras](#extras)
8. [Getting started](#getting-started)
    1. [Edit & run notebooks](#edit--run-notebooks)
    2. [See notebooks in jupyter lab](#see-notebooks-in-jupyter-lab)
9. [Build dataset](#build-dataset)
10. [Recommendation Chatbot API](#recommendation-chatbot-api)
    1. [Deployment Diagram](#deployment-diagram)
    1. [Flow Diagram](#flow-diagram)
    2. [Install as a systemd service](#install-as-a-systemd-service)
        1. [Objetives](#objetives)
        2. [Setup](#setup)
        3. [Config file](#config-file)
    3. [Register Airflow DAG](#register-airflow-dag)
    4. [Test API](#test-api)
    5. [Reset data to start evaluation process](#reset-data-to-start-evaluation-process)
    6. [API Postman Collection](#api-postman-collection)
    7. [API Documentation](#api-documentation)
12. [References](#references)

## Requisites

* [anaconda](https://www.anaconda.com/products/individual) / [miniconda](https://docs.conda.io/en/latest/miniconda.html) / [mamba](https://github.com/mamba-org/mamba)
* [mongodb](https://www.mongodb.com)
* [chromadb](https://www.trychroma.com)
* [airflow](https://airflow.apache.org/)
* [mongosh](Optional) (Optional)
* [Studio3T](https://studio3t.com/) (Optional)
* [Postman](https://www.postman.com/) (Optional)
* 6/10GB GPU to have reasonable execution times (Optional)

## Hypothesis

* Do deep learning-based models achieve better results than non-deep learning-based models? What are the advantages and disadvantages of each approach?
* How can the cold-start problem be solved in a collaborative filtering-based recommendation approach? Any proposed solutions?


## Documents

* [Specialization: Collaborative recommendation systems](https://github.com/adrianmarino/thesis-paper/blob/master/docs/thesis/thesis.pdf)
* Thesis (In progress)

## Models

The following are the models to be compared. For more details, it is recommended to refer to the thesis document in the previous section.

*  **Memory based CF**: Baseline or reference model.
   * **KNN (Cosine Distance)**
   * User-Based.
   * Item-Based.
   * Ensemble User/Item-Based.
* **Model-Based CF**: Collaborative filter models based on neural networks.
   * **Generalized Matrix Factorization (GMF)**: User/Item embeddings dot product.
   * **Biased Generalized Matrix Factorization (B-GMF)**: User/Item embeddings dot product + user/item biases.
   * **Neural Network Matrix Factorization**: User/Item Embedding + flatten + Fully Connected.
   * **Deep Factorization Machine**
* ***Ensembles**
   * Content-based and Collaborative-based models Stacking.
   * Feature Weighted Linear Stacking.
   * Multi-Bandit approach based on beta distribution.
   * LLM's + Collaborative filtering ensemble.


## Metrics

To compare collaborative filtering models, the metrics **Mean Average Precision at k (mAP@k)** y **Normalized Discounted Cumulative Gain At K (NDCG@k)** are used. Ratings between 4 and 5 points belong to the positive class, and the rest belong to the negative class.

Other metrics used:
* FBetaScore@K
* Precision@K
* Recall@K
* RMSE


## Data

To conduct the necessary tests with both collaborative filtering (CF) and content-based (CB) approaches, we need:

* Ratings of each item (movies) by the users (CF).
* Item-specific features (CB).

Based on these requirements, the following datasets were combined:

* [MovieLens 25M Dataset](https://grouplens.org/datasets/movielens/25m/): It has practically no information about the movies, but it does have user ratings.
* [TMDB Movie Dataset](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset?select=movies_metadata.csv): It does not have personalized ratings like the previous dataset, but it has several features corresponding to the movies or items which will be necessary when training content-based models.



## Notebooks

### Data pre-processing & analysis

* [Data pre-processing](https://github.com/adrianmarino/thesis-paper/tree/master/notebooks/1_data_preprocessing.ipynb)
* [Exploratory data analysis](https://github.com/adrianmarino/thesis-paper/tree/master/notebooks/2_eda.ipynb)


### Recommendation Models

#### Evaluation

* [Recommendation Models Comparative](https://github.com/adrianmarino/thesis-paper/tree/master/notebooks/4_models_comparative.ipynb)
* Recommendation Chatbot API Evaluation
  * Ensemble using Llama 2 as content based sub model
      * [Model Evaluation Process](https://github.com/adrianmarino/thesis-paper/tree/master/notebooks/chat-bot/6_evaluation-llama2.ipynb)
      * [Model Evaluation Results](https://github.com/adrianmarino/thesis-paper/tree/master/notebooks/chat-bot/9_evaluation-llama2-results.ipynb)

  * Ensemble using Llama 3 as content based sub model
    * [Model Evaluation Process](https://github.com/adrianmarino/thesis-paper/tree/master/notebooks/chat-bot/6_evaluation-llama3.ipynb)
    * [Model Evaluation Results](https://github.com/adrianmarino/thesis-paper/tree/master/notebooks/chat-bot/10_evaluation-llama3-results.ipynb)

  * [Ensemble Comparative Analysis](https://github.com/adrianmarino/thesis-paper/tree/master/notebooks/chat-bot/8_evaluation-llama2_vs_3.ipynb)

#### Baseline

* [Random Model](https://github.com/adrianmarino/thesis-paper/tree/master/notebooks/3_random_model.ipynb)

#### Collaborative Filtering

* **Memory based**
    * [KNN User/Item/Ensemple Predictors](https://github.com/adrianmarino/thesis-paper/tree/master/notebooks/cf/1_knn.ipynb)
* **Model based**
    * **Supervised**
        * [Generalized Matrix Factorization (GMF)](https://github.com/adrianmarino/thesis-paper/tree/master/notebooks/cf/2_gmf.ipynb): Embedding's + dot product.
        * [Biased Generalized Matrix Factorization (B-GMF)](https://github.com/adrianmarino/thesis-paper/tree/master/notebooks/cf/3_biased_gmf.ipynb): Embedding's + dot product + user/item bias.
        * [Neural Network Matrix Factorization](https://github.com/adrianmarino/thesis-paper/tree/master/notebooks/cf/4_nn_mf.ipynb):  User/Item Embedding + flatten + Full Connected.
        * [Deep Factorization Machine](https://github.com/adrianmarino/thesis-paper/tree/master/notebooks/cf/5_deep_fm.ipynb)
    * **Unsupervised**
        * [Collaborative Deep Auto Encoder](https://github.com/adrianmarino/thesis-paper/tree/master/notebooks/cf/7_cf-deep-ae.ipynb)
        * [Collaborative Denoising Auto Encoder](https://github.com/adrianmarino/thesis-paper/tree/master/notebooks/cf/6_cf-denoising-ae.ipynb)
        * [Collaborative Variational Auto Encoder](https://github.com/adrianmarino/thesis-paper/tree/master/notebooks/cf/8_cf-variational-ae.ipynb)
* [Supervised Stacking Ensemble](https://github.com/adrianmarino/thesis-paper/tree/master/notebooks/cf/9_stacking.ipynb)


#### Content Based

* **User Profile**
    * [User-Item filtering model (using genres only)](https://github.com/adrianmarino/thesis-paper/tree/master/notebooks/cb/1_user-item-filtering-model.ipynb)
    * [Multi-feature user profile model](https://github.com/adrianmarino/thesis-paper/tree/master/notebooks/cb/2_multi-feature-user-profile-model.ipynb)
* **Item to Item**
    * **Sparse Auto-Encoder + Distance Weighted Mean**
        * [Movie Title Sparse Auto-Encoder](https://github.com/adrianmarino/thesis-paper/tree/master/notebooks/cb/sparse/1_title_sparse_autoencoder.ipynb)
        * [Movie Tags Sparse Auto-Encoder](https://github.com/adrianmarino/thesis-paper/tree/master/notebooks/cb/sparse/2_tags_sparse_autoencoder.ipynb)
        * [Movie Genres Sparse Auto-Encoder](https://github.com/adrianmarino/thesis-paper/tree/master/notebooks/cb/sparse/3_genres_sparse_autoencoder.ipynb)
        * [Movie Overview Sparse Auto-Encoder](https://github.com/adrianmarino/thesis-paper/tree/master/notebooks/cb/sparse/4_overview_sparse_autoencoder.ipynb)
        * [Ensemple CB recommender based on Sparse Auto-Encoder](https://github.com/adrianmarino/thesis-paper/tree/master/notebooks/cb/sparse/5_ensample_sparse_autoencoder.ipynb)
    * **Sentence Transformer + Distance Weighted Mean**
        * [Movie Title Sentence Transformer](https://github.com/adrianmarino/thesis-paper/tree/master/notebooks/cb/sentence/1_title_sentence_transformer.ipynb)
        * [Movie Tags Sentence Transformer](https://github.com/adrianmarino/thesis-paper/tree/master/notebooks/cb/sentence/2_tags_sentence_transformer.ipynb)
        * [Movie Genres Sentence Transformer](https://github.com/adrianmarino/thesis-paper/tree/master/notebooks/cb/sentence/3_genres_sentence_transformer.ipynb)
        * [Movie Overview Sentence Transformer](https://github.com/adrianmarino/thesis-paper/tree/master/notebooks/cb/sentence/4_overview_sentence_transformer.ipynb)
        * [Ensemple CB recommender based on Sentence Transformer](https://github.com/adrianmarino/thesis-paper/tree/master/notebooks/cb/sentence/5_ensample_sentence_transformer.ipynb)

#### Ensembles

* [Content-based and Collaborative based models Stacking](https://github.com/adrianmarino/thesis-paper/tree/master/notebooks/ensemble/1_stacking.ipynb)

* [Feature Weighted Linear Stacking](https://github.com/adrianmarino/thesis-paper/tree/master/notebooks/ensemble/2_fwls.ipynb)

* [K-Arm Bandit + Thompson sampling](https://github.com/adrianmarino/thesis-paper/tree/master/notebooks/ensemble/3_k_arm_bandit_thompson_sampling.ipynb)

* [Recommendation Chatbot API](#recommendation-chatbot-api)
    * [Load movie items and interactions to chatbot database](https://github.com/adrianmarino/thesis-paper/tree/master/notebooks/chat-bot/1_load-items-interractions-to-database.ipynb)
    * [Update Users and Items embeddings using DeepFM model](https://github.com/adrianmarino/thesis-paper/tree/master/notebooks/chat-bot/2_embedding-db-updater.ipynb)
    * [LLM/Collaborative Filtering recommender ensemble](https://github.com/adrianmarino/thesis-paper/tree/master/notebooks/chat-bot/5_recommender.ipynb)
    * [LLM Tests](https://github.com/adrianmarino/thesis-paper/tree/master/notebooks/chat-bot/3_llm-tests.ipynb)
    * [LLM Output Parser Tests](https://github.com/adrianmarino/thesis-paper/tree/master/notebooks/chat-bot/4_output-parser-tests.ipynb)


### Extras

* [Multi-categorical variable embedding module](https://github.com/adrianmarino/thesis-paper/tree/master/notebooks/weighted_avg_embedding_bag.ipynb)



## Getting started

### Edit & run notebooks

**Step 1**: Clone repo.

```bash
$ git clone https://github.com/adrianmarino/thesis-paper.git
$ cd thesis-paper
```

**Step 2**: Create environment.

```bash
$ conda env create -f environment.yml
```

## See notebooks in jupyter lab

**Step 1**: Enable project environment.

```bash
$ conda activate thesis
```

**Step 2**: Under the project directory boot jupyter lab.

```bash
$ jupyter lab

Jupyter Notebook 6.1.4 is running at:
http://localhost:8888/?token=45efe99607fa6......
```

**Step 3**: Go to http://localhost:8888.... as indicated in the shell output.


## Build dataset


To carry out this process, it is necessary to have **MongoDB** database engine installed and listen into `localhost:27017` which is the default host & port for a homemade installation. For more instructions see:

* [Install MongoDB Community Edition on Linux](https://www.mongodb.com/docs/manual/administration/install-on-linux)
* [Install MongoDB Community Edition on Windows](https://www.mongodb.com/docs/manual/tutorial/install-mongodb-on-windows)

Now is necessary to run the next two notebooks in order:

1. [Data pre-processing](https://github.com/adrianmarino/thesis-paper/tree/master/notebooks/1_data_preprocessing.ipynb)
2. [Exploratory data analysis](https://github.com/adrianmarino/thesis-paper/tree/master/notebooks/2_eda.ipynb)

This creates two files in `datasets` path:

* `movies.json`
* `interactions.json`

These files conform to the project dataset and are used for all notebooks.



## Recommendation Chatbot API

<div style="width: 70%; float:left">
  <p>
    A chatbot API that recommends movies based on a user's text request, their profile data, and ratings. Papers on which the chatbot was based:<br/>
    <ul>
      <li><a href="https://github.com/adrianmarino/thesis-paper/blob/master/docs/ideas/2303.14524.pdf">Chat-REC: Towards Interactive and Explainable
    LLMs-Augmented Recommender System</a></li>
      <li><a href="https://github.com/adrianmarino/thesis-paper/blob/master/docs/ideas/3583780.3614949.pdf">Large Language Models as Zero-Shot Conversational
    Recommenders</a></li>
      <li><a href="https://github.com/adrianmarino/thesis-paper/blob/master/docs/ideas/3604915.3608845.pdf">Large Language Models are Competitive Near Cold-start
    Recommenders for Language- and Item-based Preferences</a></li>
    </ul>
  </p>
</div>
<div style="width: 30%; float:right">
    <p align="center">
      <img src="https://github.com/adrianmarino/thesis-paper/blob/master/images/chatbot.png?raw=true"  style="width: 300px; height: auto;" />
    </p>
</div>


### Deployment Diagram

<a href="[https://github.com/adrianmarino/thesis-paper/blob/master/docs/ideas/3583780.3614949.pdf](https://github.com/adrianmarino/thesis-paper/tree/master/docs/diagrams/recommendation-chatbot-deployment.drawio)">
<div style="width: 100%; float:right">
    <p align="center">
      <img src="https://github.com/adrianmarino/thesis-paper/blob/master/docs/diagrams/recommendation-chatbot-deployment.drawio.png?raw=true"  style="width: 100%; height: auto;" />
    </p>
</div>
</a>

### Flow Diagram

<a href="[https://github.com/adrianmarino/thesis-paper/blob/master/docs/ideas/3583780.3614949.pdf](https://github.com/adrianmarino/thesis-paper/tree/master/docs/diagrams/recommendation-chatbot-interaction.png)">
<div style="width: 100%; float:right">
    <p align="center">
      <img src="https://github.com/adrianmarino/thesis-paper/blob/master/docs/diagrams/recommendation-chatbot-interaction.png?raw=true"  style="width: 100%; height: auto;" />
    </p>
</div>
</a>


### Install as systemd service

#### Objetives
* Install `cha-bot-api` as a `systemd` daemon.
* Run daemon with your regular user.

**Note**: `systemd` is an initialization and service management system for Unix-like operating systems. It is responsible for starting the system and managing the running processes and services. `systemd` has replaced traditional initialization systems like `SysV init` in many Linux distributions due to its greater efficiency and advanced features.


#### Setup


**Step 1**: Copy `chat-bot-api.service` to user `system` config path:

```bash
$ cp chat-bot-api/chat-bot-api.service ~/.config/systemd/user/
```

**Step 2**: Refresh `systemd` daemon with updated config.

```bash
$ systemctl --user daemon-reload
```

**Step 3**: Start `chat-bot-api` daemon on boot.

```bash
$ systemctl --user enable chat-bot-api
```

**Step 6**: Start `chat-bot-api` as `systemd` daemon.

```bash
$ systemctl --user start chat-bot-api
```


**Step 7**: Check `chat-bot-api` health.

```bash
$ chat-bot-api/bin/./health
```

```json
{
   "airflow" : {
      "metadatabase" : true,
      "scheduler" : true
   },
   "chatbot_api" : true,
   "ollama_api" : true,
   "choma_database" : true,
   "mongo_database" : true
}
```


### Config file

`config.conf`:
```bash
# -----------------------------------------------------------------------------
# Python
# -----------------------------------------------------------------------------
CONDA_PATH="/opt/miniconda3"
CONDA_ENV="thesis"
# -----------------------------------------------------------------------------
#
#
#
# -----------------------------------------------------------------------------
# API
# -----------------------------------------------------------------------------
HOME_PATH="$(pwd)"
PARENT_PATH="$(dirname "$HOME_PATH")"
SERVICE_NAME="Recommendation ChatBot API"
PROCESS_NAME="uvicorn"
export API_HOST="0.0.0.0"
export API_PORT="8080"
# -----------------------------------------------------------------------------
#
#
#
# -----------------------------------------------------------------------------
# Mongo DB
# -----------------------------------------------------------------------------
export MONGODB_DATABASE="chatbot"
export MONGODB_HOST="0.0.0.0"
export MONGODB_PORT="27017"
export MONGODB_URL="mongodb://$MONGODB_HOST:$MONGODB_PORT"
# -----------------------------------------------------------------------------
#
#
#
# -----------------------------------------------------------------------------
# Chroma DB
# -----------------------------------------------------------------------------
export CHROMA_HOST="0.0.0.0"
export CHROMA_PORT="9090"
# -----------------------------------------------------------------------------
#
#
#
# -----------------------------------------------------------------------------
# Training Jobs
# -----------------------------------------------------------------------------
export TMP_PATH="$PARENT_PATH/tmp"
export DATASET_PATH="$PARENT_PATH/datasets"
export WEIGHTS_PATH="$PARENT_PATH/weights"
export METRICS_PATH="$PARENT_PATH/metrics"
# -----------------------------------------------------------------------------
#
#
#
```

### Register Airflow DAG

```bash
cp dags/cf_emb_update_dag.py $AIRFLOW_HOME/dags
```

### Test API


**Step 1**: Create a user profile.

```bash
curl --location 'http://nonosoft.ddns.net:8080/api/v1/profiles' \
--header 'Content-Type: application/json' \
--data-raw '{
    "name": "Adrian",
    "email": "adrianmarino@gmail.com",
    "metadata": {
        "studies"        : "Engineering",
        "age"            : 42,
        "genre"          : "Male",
        "nationality"    : "Argentina",
        "work"           : "Software Engineer",
        "prefered_movies": {
            "release": {
                "from" : "1970"
            },
            "genres": [
                "thiller",
                "suspense",
                "science fiction",
                "love",
                "comedy"
            ]
        }
    }
}'
```


**Step 2**: Query supported `llm`models.


```bash
curl --location 'http://nonosoft.ddns.net:8080/api/v1/recommendations/models'
```

```json
{
    "models": [
        "phi3:mini",
        "llama3-rec:latest",
        "mxbai-embed-large:latest",
        "snowflake-arctic-embed:latest",
        "llama3:text",
        "llama3:instruct",
        "llama3-8b-instruct:latest",
        "mistral:latest",
        "gemma-7b:latest",
        "gemma:7b",
        "llama2-7b-chat:latest",
        "mistral-instruct:latest",
        "mistral:instruct",
        "mixtral:latest",
        "llama2:7b-chat"
    ]
}
```


**Step 2**: Ask for recommendations.


```bash
curl --location 'http://nonosoft.ddns.net:8080/api/v1/recommendations' \
--header 'Content-Type: application/json' \
--data-raw '{
    "message": {
        "author": "adrianmarino@gmail.com",
        "content": "I want to see marvel movies"
    },
    "settings": {
        "llm"                                   : "llama3:instruct",
                                                  // llama2:7b-chat, mistral:instruct
        "retry"                                 : 3,
        "plain"                                 : false,
        "include_metadata"                      : true,
        "rag": {
            "shuffle"                           : false,
            "candidates_limit"                  : 30,
            "llm_response_limit"                : 30,
            "recommendations_limit"             : 5,
            "similar_items_augmentation_limit"  : 5,
            "not_seen"                          : true
        },
        "collaborative_filtering": {
            "shuffle"                           : false,
            "candidates_limit"                  : 100,
            "llm_response_limit"                : 30,
            "recommendations_limit"             : 5,
            "similar_items_augmentation_limit"  : 2,
            "text_query_limit"                  : 5000,
            "k_sim_users"                       : 10,
            "random_selection_items_by_user"    : 0.5,
            "max_items_by_user"                 : 10,
            "min_rating_by_user"                : 3.5,
            "not_seen"                          : true,
            "rank_criterion"                    : "user_sim_weighted_pred_rating_score"
                                                // user_sim_weighted_rating_score
                                                // user_item_sim
                                                // pred_user_rating
        }
    }
}'
```

```json
{
    "items": [
        {
            "title": "Thor",
            "poster": "http://image.tmdb.org/t/p/w500/pIkRyD18kl4FhoCNQuWxWu5cBLM.jpg",
            "release": "2011",
            "description": "Chris hemsworth stars as the norse god of thunder, who must reclaim his rightful place on the throne and defeat an evil nemesis.",
            "genres": [
                "action",
                "adventure",
                "drama",
                "fantasy",
                "imax"
            ],
            "votes": [
                "http://nonosoft.ddns.net:8080/api/v1/interactions/make/adrianmarino@gmail.com/86332/1",
                "http://nonosoft.ddns.net:8080/api/v1/interactions/make/adrianmarino@gmail.com/86332/2",
                "http://nonosoft.ddns.net:8080/api/v1/interactions/make/adrianmarino@gmail.com/86332/3",
                "http://nonosoft.ddns.net:8080/api/v1/interactions/make/adrianmarino@gmail.com/86332/4",
                "http://nonosoft.ddns.net:8080/api/v1/interactions/make/adrianmarino@gmail.com/86332/5"
            ]
        },
        {
            "title": "Avengers, The",
            "poster": "http://image.tmdb.org/t/p/w500/RYMX2wcKCBAr24UyPD7xwmjaTn.jpg",
            "release": "2012",
            "description": "Earth's mightiest heroes team up to save the world from an alien invasion in this epic superhero movie.",
            "genres": [
                "action",
                "adventure",
                "sci-fi",
                "imax"
            ],
            "votes": [
                "http://nonosoft.ddns.net:8080/api/v1/interactions/make/adrianmarino@gmail.com/89745/1",
                "http://nonosoft.ddns.net:8080/api/v1/interactions/make/adrianmarino@gmail.com/89745/2",
                "http://nonosoft.ddns.net:8080/api/v1/interactions/make/adrianmarino@gmail.com/89745/3",
                "http://nonosoft.ddns.net:8080/api/v1/interactions/make/adrianmarino@gmail.com/89745/4",
                "http://nonosoft.ddns.net:8080/api/v1/interactions/make/adrianmarino@gmail.com/89745/5"
            ]
        },
        {
            "title": "Marvel One-Shot: A Funny Thing Happened on the Way to Thor's Hammer",
            "poster": "http://image.tmdb.org/t/p/w500/njrOqsmFH4pxBrhcoslqLfw2OGk.jpg",
            "release": "2011",
            "description": "Chris hemsworth stars as the norse god of thunder, who must reclaim his rightful place on the throne and defeat an evil nemesis.",
            "genres": [
                "fantasy",
                "sci-fi"
            ],
            "votes": [
                "http://nonosoft.ddns.net:8080/api/v1/interactions/make/adrianmarino@gmail.com/168040/1",
                "http://nonosoft.ddns.net:8080/api/v1/interactions/make/adrianmarino@gmail.com/168040/2",
                "http://nonosoft.ddns.net:8080/api/v1/interactions/make/adrianmarino@gmail.com/168040/3",
                "http://nonosoft.ddns.net:8080/api/v1/interactions/make/adrianmarino@gmail.com/168040/4",
                "http://nonosoft.ddns.net:8080/api/v1/interactions/make/adrianmarino@gmail.com/168040/5"
            ]
        }
    ]
}
```

### Reset data to start evaluation process


**Step 1**: Backup user interactions in `mongodb`.

```bash
mongoexport -d chatbot -c interactions --out interactions.json --jsonArray 
```

**Step 2**: Remove all chatbot user interactions in `mongodb`.

```javascript
db.getCollection('interactions').deleteMany({ 'user_id': { $regex: /@/ }})
```

**Step 3**: Backup and Remove all users profiles in `mongodb`.

```bash
mongoexport -d chatbot -c profiles --out profiles.json --jsonArray 
```

```javascript
db.getCollection('profiles').drop();
```

**Step 4**: Backup and Remove all predicted interactions in `mongodb`.

```bash
mongoexport -d chatbot -c pred_interactions --out pre_interactions.json --jsonArray
```

```javascript
db.getCollection('pred_interactions').drop();
```

**Step 5**: Remove users search history in `mongodb`.

```javascript
db.getCollection('histories').drop();
```

**Step 6**: Remove all collections in `chroma` database.

```bash
cd chat-bot-api
bin/./chroma-delete-all

ENV: thesis
2024-06-08 13:31:53,826 - INFO - Start: Delete all chroma db collections...
2024-06-08 13:31:58,376 - INFO - ==> "items_cf" collection deleted...
2024-06-08 13:31:58,685 - INFO - ==> "items_content" collection deleted...
2024-06-08 13:31:59,130 - INFO - ==> "users_cf" collection deleted...
2024-06-08 13:31:59,130 - INFO - Finish: 3 collections deleted
```

**Step 7**: Restart charbot API.

```bash
systemctl --user restart chat-bot-api
```

**Step 8**: Rebuild item text embeddings used to search items by free text (Retrieval Augmented Generation).

```bash
curl --location --request PUT 'http://nonosoft.ddns.net:8080/api/v1/items/embeddings/content/build?batch_size=5000'
```

Could use next command to see reindex process logs:

```bash
tail -f /var/tmp/chat-bot-api.log
```

**Step 9**: Restart `chat-bot-api`

```bash
systemctl --user restart chat-bot-api
```

```bash
systemctl --user status chat-bot-api

● chat-bot-api.service - Recommendation Chatbot API for adrian user
     Loaded: loaded (/home/adrian/.config/systemd/user/chat-bot-api.service; enabled; preset: enabled)
     Active: active (exited) since Sat 2024-06-08 13:35:12 -03; 3s ago
    Process: 4092833 ExecStart=/home/adrian/chat-bot-api/bin/start (code=exited, status=0/SUCCESS)
   Main PID: 4092833 (code=exited, status=0/SUCCESS)
      Tasks: 26 (limit: 38212)
     Memory: 514.4M (peak: 515.0M)
        CPU: 4.855s
     CGroup: /user.slice/user-1000.slice/user@1000.service/app.slice/chat-bot-api.service
             ├─4092894 python -m uvicorn api:app --reload --host 0.0.0.0 --port 8080
             ├─4092897 /home/adrian/.conda/envs/thesis/bin/python -c "from multiprocessing.resource_tracker import main;main(4)"
             └─4092898 /home/adrian/.conda/envs/thesis/bin/python -c "from multiprocessing.spawn import spawn_main; spawn_main(tracker_fd=5, pipe_handle=7)" --multiprocessing-fork

jun 08 13:35:12 skynet systemd[1467]: Starting Recommendation Chatbot API for adrian user...
jun 08 13:35:12 skynet start[4092833]: ENV: thesis
jun 08 13:35:12 skynet start[4092833]: Start Recommendation ChatBot API...
jun 08 13:35:12 skynet systemd[1467]: Finished Recommendation Chatbot API for adrian user.
jun 08 13:35:12 skynet start[4092894]: INFO:     Will watch for changes in these directories: ['/home/adrian/development/personal/maestria/thesis-paper/chat-bot-api']
jun 08 13:35:12 skynet start[4092894]: INFO:     Uvicorn running on http://0.0.0.0:8080 (Press CTRL+C to quit)
jun 08 13:35:12 skynet start[4092894]: INFO:     Started reloader process [4092894] using WatchFiles
```

**Step 10**: Remove previos model weights.

```bash
rm -rf weights
```

**Step 11**: Start Jupyter Lab, go to `notebooks/chat-bot/6_evaluation-llama3.ipynb` and start notebook.

```bash
cd ..
jupyterlab
```

**Note**: The evaluation process takes between 4 to 5 days.


### API Postman Collection

* [Recommendation Charbot API postman collection](https://github.com/adrianmarino/thesis-paper/blob/master/chat-bot-api/postman_collection.json)
* [How to import a Postman collection](https://www.youtube.com/watch?v=M-qHvBhULes)


### API Documentation

* [Swagger UI](http://nonosoft.ddns.net:8080/docs)
* [Redoc](http://nonosoft.ddns.net:8080/redoc)


## References
   * [References](https://github.com/adrianmarino/thesis-paper/tree/master/notebooks/5_references.ipynb)
   * Using or based on
      * [pytorch-common](https://github.com/adrianmarino/pytorch-common)
      * [knn-cf-rec-sys](https://github.com/adrianmarino/knn-cf-rec-sys)
      * [deep-fm](https://github.com/adrianmarino/deep-fm)
      * [recommendation-system-approaches](https://github.com/adrianmarino/recommendation-system-approaches)
