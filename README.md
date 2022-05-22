
# Thesis-papes

La idea principal es comparar distintos enfoques para construir modelos de recomendación basados en filtros colaborativos e híbridos (es decir,  una combinación entre filtros colaborativos y basados en contenido), explicando ventajas y desventajas de cada enfoque y la arquitectura y funcionamiento de cada modelo.

## Modelos

A continuacion se especifican los modelos a comaprar:

 * Memory based CF: Sera el baseline o modelo de referencia, del cual queremos obtener mejores resultados.
     * [KNN usando distintas distancias (User-User y Item-Item)](https://github.com/adrianmarino/knn-cf-rec-sys). **Done**
 * Model Based CF: Modelos de filtros colaborativos basados en redes neuronales.
   * Collaborative Filtering
      * Generalized Matrix Factorization (GMF)
        * Embedding's + dot product.
        * Embedding's + dot product + user bias + item bias.
      * Neural Network Matrix Factorization: Embedding's + Full Connected Layers.
      * [Deep Matrix Factorization ](https://arxiv.org/pdf/1703.04247.pdf) **Done**
    * Hybrid CB + CF model: Combinan filtros colaborativos con el enfoque basado en contenido.
        * Neural Network Matrix Factorization: Embedding's + Full Connected Layers.
        * [Deep Matrix Factorization ](https://arxiv.org/pdf/1703.04247.pdf)

## Metricas

Estyo utilizando la metrica **Mean Average Precision at k (mAP@k)**. Dada una lista de k items ordenados desc. por ratings predicho para el usuario; esta metrica, permite medir la frecuencia con que la que se encuentram X ratings en las primeras posiciones en una lsita de items recomendados. Por ejemplo: ratings entre 4 y 5.

Otras métricas utilizadas:
* RMSE
* FBetaScore
* Precision@K
* Recall@K
* FBetaScore@K

## Hipótesis

Intentó determinar si modelos basados en deep learning obtienen mejores resultados a modelos que no estan basado en deep learning y cuales son las ventajas y desventajas a la hora de implementar cada enfoque. Los modelos a comparar son distintas implementaciones de filtros colaborativos y modelos hibridos (CB + CF).

## Datos

Para tener los datos necesario para probar los enfoques de filtros colaborativos(CF) y basados en contenido(CB) necesito:
* Calificaciones de los ítems(movies) de los usuarios (CF)
* Features propies de los ítems (CB)

Dado esto, combine los siguientes datasets:

* [MovieLens 25M Dataset](https://grouplens.org/datasets/movielens/25m/): Prácticamente no tiene información de las películas pero si tiene las calificaciones de los usuarios.
* [TMDB Movie Dataset](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset?select=movies_metadata.csv): No tiene calificaciones personalizadas como el dataset anterior pero tiene varios features para las películas que es lo que necesito.


## Referencias

[References](https://github.com/adrianmarino/thesis-paper/tree/master/notebooks/references.ipynb)


## Extras

Dada la demanda de procesamiento que tienen estos modelos estoy implementado todo en [pytorch](https://pytorch.org) para poder usar GPU, ya que [scikit-learn](https://scikit-learn.org/stable/) no lo permite. También estuve trabajando en un modelo de capas para poder pasar como input las variables overview y tags de las películas ([Multi-categorical variable embedding module](https://github.com/adrianmarino/thesis-paper/tree/master/notebooks/weighted_avg_embedding_bag.ipynb)).

## Notebooks

* [Preprocesamiento](https://github.com/adrianmarino/thesis-paper/tree/master/notebooks/preprocessing-integration.ipynb)
* [Analisis exploratorio](https://github.com/adrianmarino/thesis-paper/tree/master/notebooks/eda.ipynb)
* Memory based CF
  * [KNN CF Model (User-User/Item-Item)](https://github.com/adrianmarino/thesis-paper/tree/master/notebooks/models_knn.ipynb)
* Model based CF
  * [Deep FM CF](https://github.com/adrianmarino/thesis-paper/tree/master/notebooks/models_deep_fm.ipynb)
* [Multi-categorical variable embedding module](https://github.com/adrianmarino/thesis-paper/tree/master/notebooks/weighted_avg_embedding_bag.ipynb)

## Using or based on

* [pytorch-common](https://github.com/adrianmarino/pytorch-common)
* [knn-cf-rec-sys](https://github.com/adrianmarino/knn-cf-rec-sys)
* [deep-fm](https://github.com/adrianmarino/deep-fm)


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

