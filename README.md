[![CircleCI](https://dl.circleci.com/status-badge/img/gh/adrianmarino/thesis-paper/tree/master.svg?style=shield)](https://dl.circleci.com/status-badge/redirect/gh/adrianmarino/thesis-paper/tree/master)   ![Coverage](https://raw.githubusercontent.com/adrianmarino/thesis-paper/master/coverage.svg)

# UBA - Maestria en Explotación de Datos y Descubrimiento de Conocimiento - Thesis - Sistemas de Recomendación Colaborativos e Híbridos 

Este trabajo busca realizar una comparativa de distintos enfoques de recomendación basados en filtros colaborativos e híbridos (Es decir, una combinación de filtros colaborativos y basados en contenido), explicando ventajas y desventajas de cada enfoque, su arquitectura y funcionamiento para cada modelo propuesto.
## Thesis

[Thesis (In progress)](https://github.com/adrianmarino/thesis-paper/blob/master/docs/thesis/thesis.pdf)


## Modelos

A continuación se especifican los modelos a comparar.para moas detalle de los mismo se recomienda ver el documento de tesis del apartado anterior.

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
     
 * **Enfoque Híbrido**: Combinación de filtros colaborativos(CF) con enfoque basado en contenido(CB). Esto permite lidiar con el problema de cold-start de CF.
    * **Modelos Híbrido 1: Modelo de CF + Sparse Auto-Encoder (CB)**
        * Si el usuario tiene menos de 20 interacciones:
            1. Se entrena un modelo de CF y generamos los embedding de usuarios e items.
            2. Sparse Auto-Encoder: Se entrena un Auto-Encoder para cada variable tipo texto (title, genres, tags y overview). Como resultado tenemos embeddings para cada variable.
            3. Finalmente, se genera una lista de recomendaciones para un item promediando las distancias coseno con todos los embeddings: el embedding de items y todo los embeddings de variables tipo texto generados con los Auto-Encoders.
            4. Se realiza un promedio pesado por cada variable para poder controlar cuanto influye cada una en el ranking final.
        * Cuando el usuario tiene mas de 20 interacciones:
            * Se usa el modelo de CF. 
            * Lo idea seria mezclar recomendaciones de ambos modelos, ya que CF puede tener recomendaciones muy personalizadas o long tail. 
    * **Modelos Híbrido 2: Modelo CF + Sentence Transformer**
        * Es la misma idea que el modelo 1, pero se utiliza un modelo de lenguaje Sentence Transformer en vez de un Sparse Auto-Encoder, el cual genera un embedding para cada frase de texto para variables tipo texto.     
 * **Ensample/Stacking de modelos**

## Métricas

Para comparar los modelos basados en filtros colaborativos se utilizan las métricas **Mean Average Precision at k (mAP@k)** y **Normalized Discounted cumulative gain At K (NDCG@k)**. Dada una lista de k ítems ordenados desc. por ratings predicho para el usuario; esta métrica, permite medir la frecuencia con que la que se encuentran X ratings en las primeras posiciones en una lista de ítems recomendados. Por ejemplo: ratings entre 4 y 5.

Otras métricas utilizadas:

* NDCG@k
* FBetaScore@K
* Precision@K
* Recall@K
* RMSE

## Hipótesis

* ¿Los modelos basados en deep learning obtienen mejores resultados que modelos no basados en deep learning? ¿Cuáles son las ventajas y desventajas de cada enfoque?
* ¿Cómo se puede solucionar el problema de cold-start que sufre el enfoque de recomendación basado en filtros colaborativos? ¿Propuestas de solución?

## Datos

Para poder realizar las pruebas necesarias con ambos enfoques (filtros colaborativos(CF) y basados en contenido(CB)) necesitamos:

* Calificaciones de cada ítems(movies) por parte de los usuarios (CF).
* Features propios de los ítems (CB).

Dadas estas necesidades, se combinaron los siguientes datasets:

* [MovieLens 25M Dataset](https://grouplens.org/datasets/movielens/25m/): Prácticamente no tiene información de las películas pero si tiene las calificaciones de los usuarios.
* [TMDB Movie Dataset](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset?select=movies_metadata.csv): No tiene calificaciones personalizadas como el \textit{dataset} anterior, pero tiene varios features corrspondiente a las películas o items los cuales seran necesarios cunado se entrenen modelos basados en contenido.


## Referencias

[References](https://github.com/adrianmarino/thesis-paper/tree/master/notebooks/references.ipynb)


## Notebooks

* [Data pre-processing](https://github.com/adrianmarino/thesis-paper/tree/master/notebooks/1_data-preprocessing.ipynb)
* [Exploratory data analysis](https://github.com/adrianmarino/thesis-paper/tree/master/notebooks/2_eda.ipynb)

* **Models**    
    * **Collaborative Filtering**
      * [Comparatives (Supervised)](https://github.com/adrianmarino/thesis-paper/tree/master/notebooks/cf/10_comparatives.ipynb)
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

* [Ensemple CB recommender based on Sentence Transformer](https://github.com/adrianmarino/thesis-paper/tree/master/notebooks/models_movie_ensample_sentence_transformer.ipynb)
    * **Hybrid Models**
        * **CF + Sparse Auto-Encoder** (Pending)
        * **CF + Sentence Transformer** (Pending)

* [Multi-categorical variable embedding module](https://github.com/adrianmarino/thesis-paper/tree/master/notebooks/weighted_avg_embedding_bag.ipynb)


## Using or based on

* [pytorch-common](https://github.com/adrianmarino/pytorch-common)
* [knn-cf-rec-sys](https://github.com/adrianmarino/knn-cf-rec-sys)
* [deep-fm](https://github.com/adrianmarino/deep-fm)
* [recommendation-system-approaches](https://github.com/adrianmarino/recommendation-system-approaches)

## Requisites

* [anaconda](https://www.anaconda.com/products/individual) / [miniconda](https://docs.conda.io/en/latest/miniconda.html)


## Getting started

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

**Step 2**: Under project directory boot jupyter lab.

```bash
$ jupyter lab

Jupyter Notebook 6.1.4 is running at:
http://localhost:8888/?token=45efe99607fa6......
```

**Step 3**: Go to http://localhost:8888.... as indicated in the shell output.

