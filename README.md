
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
    1. [Arquitecture Diagram](#arquitecture-diagram)
    2. [Setup as a systemd service](#setup-as-a-systemd-service)
        1. [Objetives](#objetives)
        2. [Setup](#setup)
        3. [Config file](#config-file)
    3. [Register Airflow DAG](#register-airflow-dag)
    4. [Test API](#test-api)
    5. [API Postman Collection](#api-postman-collection)
    6. [API Documentation](#api-documentation)
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
  * [Using Llama 2 as content based sub model](https://github.com/adrianmarino/thesis-paper/tree/master/notebooks/chat-bot/6_evaluation-llama2.ipynb)
  * [Using Llama 3 as content based sub model](https://github.com/adrianmarino/thesis-paper/tree/master/notebooks/chat-bot/6_evaluation-llama3.ipynb)
  * [Llama 3 vs. Llama 2 as content based sub model](https://github.com/adrianmarino/thesis-paper/tree/master/notebooks/chat-bot/8_evaluation-llama2_vs_3.ipynb)

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
        * [Movie Title Sparse Autoencoder](https://github.com/adrianmarino/thesis-paper/tree/master/notebooks/cb/sparse/1_title_sparse_autoencoder.ipynb)
        * [Movie Tags Sparse Autoencoder](https://github.com/adrianmarino/thesis-paper/tree/master/notebooks/cb/sparse/2_tags_sparse_autoencoder.ipynb)
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

### Arquitecture Diagram

<a href="[https://github.com/adrianmarino/thesis-paper/blob/master/docs/ideas/3583780.3614949.pdf](https://github.com/adrianmarino/thesis-paper/tree/master/docs/diagrams/recommendation-chatbot.drawio.png)">
<div style="width: 100%; float:right">
    <p align="center">
      <img src="https://github.com/adrianmarino/thesis-paper/blob/master/docs/diagrams/recommendation-chatbot.drawio.png?raw=true"  style="width: 100%; height: auto;" />
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
        "llama2-13b-chat",
        "llama2-7b-chat",
        "gemma-7b",
        "mistral-instruct",
        "mistral",
        "neural-chat"
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
