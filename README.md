
# UBA - Maestria en Explotación de Datos y Descubrimiento de Conocimiento - Thesis - Sistemas de Recomendación Colaborativos e Híbridos 

Este trabajo busca realizar una comparativa de distintos enfoques de recomendación basados en filtros colaborativos e híbridos (Es decir, una combinación de filtros colaborativos y basados en contenido), explicando ventajas y desventajas de cada enfoque, su arquitectura y funcionamiento para cada modelo propuesto.

## Thesis

[Thesis (In progress)](https://github.com/adrianmarino/thesis-paper/blob/master/docs/thesis/thesis.pdf)


## Modelos

A continuacion se especifican los modelos a comparar:

 *  **Memory based CF**: Bseline o modelo de referencia.
    * **KNN (Distancia Coseno)**
      * User Based.
      * Item Based.
      * Ensample User/Item Based.
 
 *  **Model Based CF**: Modelos de filtros colaborativos basados en redes neuronales.
    *   **Generalized Matrix Factorization (GMF)**: User/Item embeddings dot product.
    *   **Biased Generalized Matrix Factorization (B-GMF)**: User/Item embeddings dot product + user/item biases.
    *   **Neural Network Matrix Factorization**: User/Item Embedding + flatten + Full Connected.
    *   **Deep Factorization Machine**
     
 * **Enfoque Híbrido**: Combinacion de filtros colaborativos con enfoque basado en contenido(CB). Esto permite lidiar con el problema de cold-start de CF.
    * **Modelos Hibrido 1: Modelo CF + Sparce Autoencoder**
        * Si el usaurio tiene menso de 20 inreracciones:
            1. Se entrena un modelo de CF y generamos los embeddins de usuario e items.
            2. Sparse auto-encoder: Se entrena un autoencoder para cada variable tipo texto(title, genery, tags y overview). Como resultado tenemos embedddinds para cada variable.
            3. Finalmente se genera una lista de recomendaciones para un item promediando las distancias coseno con todos los embeddindgs: el embedding de items y todo los embeddins de variables tipo texto.
            4. Se realiza un promedio pesado por cada veriable apra poder controlar cuando influye cada una en el rankeo final.
        * Si el usaurio tiene mas de 20 interacciones
            * Se usael modelo de CF. 
            * Lo idea seria mezclar recomendaciones de ambos modelos ya que FC puede tener recomendaciones muy personalizadas o long tail. 
    * **Modelos Hibrido 2: Modelo CF + Sentence Transformer**
        * Es la misma idea que el modelo 1, pero se utiliza un modelo de lenguaje Sentence Transformer en vez de un Sparce autoencoder, el cual genera un embedding para cada frase de texto para variables tipo texto.     
 * **Ensample/Stacking de modelos**

## Metricas

Para comparar los modelos basados en filtros colaborativos se utiliza la métrica **Mean Average Precision at k (mAP@k)**. Dada una lista de k ítems ordenados desc. por ratings predicho para el usuario; esta métrica, permite medir la frecuencia con que la que se encuentran X ratings en las primeras posiciones en una lista de ítems recomendados. Por ejemplo: ratings entre 4 y 5.

Otras métricas utilizadas:

* RMSE
* FBetaScore@K
* Precision@K
* Recall@K

## Hipótesis

* ¿Los modelos basados en deep learning obtienen mejores resultados que modelos no basados en deep learning? ¿Cuáles son las ventajas y desventajas de cada enfoque?
* ¿Cómo se puede solucionar el problema de cold-start que sufre el enfoque de recomendación basado en filtros colaborativos? ¿Propuestas de solución?

## Datos

Para poder realizar las pruebas necesarias con ambios enfoques (filtros colaborativos(CF) y basados en contenido(CB)) necesitamos:

* Calificaciones de cada ítems(movies) por parte de los usuarios (CF).
* Features propios de los ítems (CB).

Dadas estas necesidades, se combinaron los siguientes datasets:

* [MovieLens 25M Dataset](https://grouplens.org/datasets/movielens/25m/): Prácticamente no tiene información de las películas pero si tiene las calificaciones de los usuarios.
* [TMDB Movie Dataset](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset?select=movies_metadata.csv): No tiene calificaciones personalizadas como el dataset anterior pero tiene varios features para las películas que es lo que necesito.


## Referencias

[References](https://github.com/adrianmarino/thesis-paper/tree/master/notebooks/references.ipynb)


## Notebooks

* [Preprocesamiento](https://github.com/adrianmarino/thesis-paper/tree/master/notebooks/data-preprocessing.ipynb)
* [Analisis exploratorio](https://github.com/adrianmarino/thesis-paper/tree/master/notebooks/eda.ipynb)

* Models
    * [Model Comparatives](https://github.com/adrianmarino/thesis-paper/tree/master/notebooks/models_comparatives.ipynb)
    
    * **Collaborative Filtering**
        * **Memory based**
          * [KNN User/Item/Ensemple Predictors](https://github.com/adrianmarino/thesis-paper/tree/master/notebooks/models_knn.ipynb): Usando distancia coseno y distacia coseno ajustada.
        * **Model based**
          * [Generalized Matrix Factorization (GMF)](https://github.com/adrianmarino/thesis-paper/tree/master/notebooks/models_gmf.ipynb): Embedding's + dot product.
          * [Biased Generalized Matrix Factorization (B-GMF)](https://github.com/adrianmarino/thesis-paper/tree/master/notebooks/models_gmf_bias.ipynb): Embedding's + dot product + user/item bias.
          * [Neural Network Matrix Factorization](https://github.com/adrianmarino/thesis-paper/tree/master/notebooks/models_nn_mf.ipynb):  User/Item Embedding + flatten + Full Connected.
          * [Deep Factorization Machine](https://github.com/adrianmarino/thesis-paper/tree/master/notebooks/models_deep_fm.ipynb)
    * **Content Based**
       * **Sparse auto-encoder + promedio pesado**
           * [Movie Tags Sparse Autoencoder](https://github.com/adrianmarino/thesis-paper/tree/master/notebooks/models_movie_tags_sparse_autoencoder.ipynb)
           * [Movie Overview Sparse Autoencoder](https://github.com/adrianmarino/thesis-paper/tree/master/notebooks/models_movie_overview_sparse_autoencoder.ipynb)
           * [Movie Genres Sparse Autoencoder](https://github.com/adrianmarino/thesis-paper/tree/master/notebooks/models_movie_genres_sparse_autoencoder.ipynb)
           * [Ensemple CB recommender based on Sparse Autoencoder](https://github.com/adrianmarino/thesis-paper/tree/master/notebooks/models_movie_ensample_sparse_autoencoder.ipynb)
       * **Sentence Transformer + promedio pesado**
           * [Movie Title Sentence Transformer](https://github.com/adrianmarino/thesis-paper/tree/master/notebooks/models_movie_title_sentence_transformer.ipynb)
           * [Movie Overview Sentence Transformer](https://github.com/adrianmarino/thesis-paper/tree/master/notebooks/models_movie_overview_sentence_transformer.ipynb)
           * [Movie Genres Sentence Transformer](https://github.com/adrianmarino/thesis-paper/tree/master/notebooks/models_movie_genres_sentence_transformer.ipynb)
           * [Movie Tags Sentence Transformer](https://github.com/adrianmarino/thesis-paper/tree/master/notebooks/models_movie_tags_sentence_transformer.ipynb)
           * [Ensemple CB recommender based on Sentence Transformer](https://github.com/adrianmarino/thesis-paper/tree/master/notebooks/models_movie_ensample_sentence_transformer.ipynb)
    * Enfoque Híbrido
        * **Modelos Hibrido 1: Modelo CF + Sparce Autoencoder** (Pending)
        * **Modelos Hibrido 2: Modelo CF + Sentence Transformer** (Pending)
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

