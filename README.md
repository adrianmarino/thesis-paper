
# Thesis-papes

* La idea principal es comparar RecSys basado en filtros colaborativos y enfoques hibridos (CB+CF) para lidiar cold-start scenarios:
    * Memory based CF
        * KNN usando distintas distancias (User-User y Item-Item). **Done**
    * Model Based CF
        * DeepFM model. **Done**
        * Embedding + dense model.
        * Embedding + dot + u/m bias model.
    * Hybrid CB + CF model.
        * DeepFM.
        * Embedding + dense model.
* Explicar la ventajas y debilidades de cada enfoque.
* Explicar la arquitectura de cada modelo.

### Notebooks

* [Preprocesamiento](https://github.com/adrianmarino/thesis-paper/tree/master/notebooks/preprocessing-integration.ipynb)
* [Analisis exploratorio](https://github.com/adrianmarino/thesis-paper/tree/master/notebooks/eda.ipynb)
* Memory based CF
  * [KNN CF Model (User-User/Item-Item)](https://github.com/adrianmarino/thesis-paper/tree/master/notebooks/models_knn.ipynb)
* Model based CF
  * [Deep FM CF](https://github.com/adrianmarino/thesis-paper/tree/master/notebooks/models_deep_fm.ipynb)
* [Multi-categorical variable embedding module](https://github.com/adrianmarino/thesis-paper/tree/master/notebooks/weighted_avg_embedding_bag.ipynb)
* [References](https://github.com/adrianmarino/thesis-paper/tree/master/notebooks/references.ipynb)


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

