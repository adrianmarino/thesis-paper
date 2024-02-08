
# UBA - Maestria en Explotación de Datos y Descubrimiento de Conocimiento - Thesis - Sistemas de Recomendación Colaborativos e Híbridos 

Este trabajo busca realizar una comparativa de distintos enfoques de recomendación basados en filtros colaborativos e híbridos (Es decir, una combinación de filtros colaborativos y basados en contenido), explicando ventajas y desventajas de cada enfoque, su arquitectura y funcionamiento para cada modelo propuesto.


## Documents

* [Specialization: Collaborative recommendation systems](https://github.com/adrianmarino/thesis-paper/blob/master/docs/thesis/thesis.pdf)
* Thesis (In progress)

## Models

A continuación se especifican los modelos a comparar. Para mas detalle se recomienda ver el documento de tesis del apartado anterior.

 *  **Memory based CF**: Baseline o modelo de referencia.
    * **KNN (Distancia Coseno)**
      * User Based.
      * Item Based.
      * Ensample User/Item Based.
 
 *  **Model Based CF**: Modelos de filtros colaborativos basados en redes neuronales.
    *   **Generalized Matrix Factorization (GMF)**: User/Item embeddings dot product.
    *   **Biased Generalized Matrix Factorization (B-GMF)**: User/Item embeddings dot product + user/item biases.
    *   **Neural Network Matrix Factorization**: User/Item Embedding + flatten + Full Connected.
    *   **Deep Factorization Machine**
 
 * **Ensamples**
   * Content-based and Collaborative based models Stacking.
   * Feature Weighted Linear Stacking.
   * Muti-Bandit approach based on beta distribution.


## Metrics

Para comparar los modelos basados en filtros colaborativos se utilizan las métricas **Mean Average Precision at k (mAP@k)** y **Normalized Discounted Cumulative Gain At K (NDCG@k)**. Ratings entre 4 y 5 puntos pertenecen a la clase positiva y restro en la clase negativa.

Otras métricas utilizadas:

* FBetaScore@K
* Precision@K
* Recall@K
* RMSE

## Hypothesis

* ¿Los modelos basados en deep learning obtienen mejores resultados que modelos no basados en deep learning? ¿Cuáles son las ventajas y desventajas de cada enfoque?
* ¿Cómo se puede solucionar el problema de cold-start que sufre el enfoque de recomendación basado en filtros colaborativos? ¿Propuestas de solución?

## Data

Para poder realizar las pruebas necesarias con ambos enfoques (filtros colaborativos(CF) y basados en contenido(CB)) necesitamos:

* Calificaciones de cada ítems(movies) por parte de los usuarios (CF).
* Features propios de los ítems (CB).

Dadas estas necesidades, se combinaron los siguientes datasets:

* [MovieLens 25M Dataset](https://grouplens.org/datasets/movielens/25m/): Prácticamente no tiene información de las películas pero si tiene las calificaciones de los usuarios.
* [TMDB Movie Dataset](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset?select=movies_metadata.csv): No tiene calificaciones personalizadas como el \textit{dataset} anterior, pero tiene varios features corrspondiente a las películas o items los cuales seran necesarios cunado se entrenen modelos basados en contenido.


## References
   * [References](https://github.com/adrianmarino/thesis-paper/tree/master/notebooks/5_references.ipynb)
   * Using or based on
      * [pytorch-common](https://github.com/adrianmarino/pytorch-common)
      * [knn-cf-rec-sys](https://github.com/adrianmarino/knn-cf-rec-sys)
      * [deep-fm](https://github.com/adrianmarino/deep-fm)
      * [recommendation-system-approaches](https://github.com/adrianmarino/recommendation-system-approaches)


## Notebooks


* **Recommendation Models**

    * [Models Comparative](https://github.com/adrianmarino/thesis-paper/tree/master/notebooks/4_models_comparative.ipynb)

    * [Random Model](https://github.com/adrianmarino/thesis-paper/tree/master/notebooks/3_random_model.ipynb)

    * **Collaborative Filtering**
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
    * **Content Based**
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

    * **Ensembles**
        * [Content-based and Collaborative based models Stacking](https://github.com/adrianmarino/thesis-paper/tree/master/notebooks/ensemble/1_stacking.ipynb)

        * [Feature Weighted Linear Stacking](https://github.com/adrianmarino/thesis-paper/tree/master/notebooks/ensemble/2_fwls.ipynb)

        * [K-Arm Bandit + Thompson sampling](https://github.com/adrianmarino/thesis-paper/tree/master/notebooks/ensemble/3_k_arm_bandit_thompson_sampling.ipynb)

        * **Recommendation ChatBot**
            * Papers on which the chatbot was based.
                * [Chat-REC: Towards Interactive and Explainable
                LLMs-Augmented Recommender System](https://github.com/adrianmarino/thesis-paper/blob/master/docs/ideas/2303.14524.pdf)
                * [Large Language Models as Zero-Shot Conversational
                Recommenders](https://github.com/adrianmarino/thesis-paper/blob/master/docs/ideas/3583780.3614949.pdf)
                * [Large Language Models are Competitive Near Cold-start
                Recommenders for Language- and Item-based Preferences](https://github.com/adrianmarino/thesis-paper/blob/master/docs/ideas/3604915.3608845.pdf)
            * [Load movie items and interactions to chatbot database](https://github.com/adrianmarino/thesis-paper/tree/master/notebooks/chat-bot/1_load-items-interractions-to-database.ipynb)
            * [Update Users and Items embeddings using DeepFM model](https://github.com/adrianmarino/thesis-paper/tree/master/notebooks/chat-bot/2_embedding-db-updater.ipynb)
            * [LLM Tests](https://github.com/adrianmarino/thesis-paper/tree/master/notebooks/chat-bot/3_llm-tests.ipynb)
            * [LLM Output Parser Tests](https://github.com/adrianmarino/thesis-paper/tree/master/notebooks/chat-bot/4_output-parser-tests.ipynb)
            * [Collaborative Filtering recommender using user embeddings from chromadb](https://github.com/adrianmarino/thesis-paper/tree/master/notebooks/chat-bot/5_recommender.ipynb)



* **Extras**
    * [Multi-categorical variable embedding module](https://github.com/adrianmarino/thesis-paper/tree/master/notebooks/weighted_avg_embedding_bag.ipynb)


## Requisites

* [anaconda](https://www.anaconda.com/products/individual) / [miniconda](https://docs.conda.io/en/latest/miniconda.html) / [mamba](https://github.com/mamba-org/mamba)
* [mongodb](https://www.mongodb.com)
* [chromadb](https://www.trychroma.com)
* [mongosh](Optional)
* [Studio3T](https://studio3t.com/) (Optional)
* 6/10GB GPU to have reasonable execution times (Optional)

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

1. [Data pre-processing](https://github.com/adrianmarino/thesis-paper/tree/master/notebooks/1_data-preprocessing.ipynb)
2. [Exploratory data analysis](https://github.com/adrianmarino/thesis-paper/tree/master/notebooks/2_eda.ipynb)

This creates two files in `datasets` path:

* `movies.json`
* `interactions.json`

These files conform to the project dataset and are used for all notebooks.

