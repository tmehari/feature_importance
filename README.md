# ECG Feature Importance Rankings: Cardiologists vs. Algorithms

This repository accompanies the papers [Multi-Class ECG Feature Importance Rankings: Cardiologists vs Algorithms](https://ieeexplore.ieee.org/abstract/document/10081737) and [ECG Feature Importance Rankings: Cardiologists vs. Algorithms](https://arxiv.org/abs/2305.17043).



## Installation

Create and activate an Anaconda environment:

```bash
conda env create -f environment.yml
conda activate fi
```

## Compute Feature Importance Rankings
### TLDR; 
To reproduce the results from [ECG Feature Importance Rankings: Cardiologists vs. Algorithms](https://arxiv.org/abs/2305.17043), simply run 

```
for i in {1..5}
do
    python code/run_binary_feature_importance_experiments.py --output_dir "output${i}"
done
```
Then, navigate through the `./scoring_algorithm.ipynb` notebook to generate all the tables presented in the study.
<br>

There are two different implementations for feature importance rankings:

1. Binary Classification: One pathology out of `[AVBlock, LBBB, RBBB]` vs `NORM`.
2. One vs All: Train a one-vs-all classification task for one pathology out of `[AVBlock, LBBB, NORM, RBBB]` vs the others.

The binary classification is implemented in `code/run_binary_feature_importance_experiments.py`, and the one-vs-all classification is implemented in `code/run_multiclass_feature_importance_experiments.py`.

We offer three model-dependent explanation methods: `[lime, shap, permutation]`, for these four models: `[random forest, logistic regression, xgb, neural network]`.

Additionally, we offer five filter methods, which are methods that do not use machine learning models at their base. Namely, these are: *Maximum Relevance - Minimum Redundancy (MRMR)*, *Neighbourhood Component Analysis (NCA)*, *Relieff*, *Modified ROCAUC*, and the *&chi;<sup>2</sup> Test*.

Some machine learning models implicitly learn feature importance values. We offer the feature importance values of three models: `[random forest, logistic regression, Gaussian Process Classifier]`, and call them the implicit feature importance methods.

Hence, in total, we offer `3*4+5+3=20` feature importance methods. 
You can simply run all experiments for all pathologies by 

```
python code/run_binary_feature_importance_experiments.py --output_dir ./path/to/output/directory
```

If you do not specify an output directory, the results will be saved to `./output`. If you are interested in a single experiment you can, for example, calculate the rankings for the binary classification task for `LBBB` vs `NORM` with `shap` as the explanation method and `logistic regression` and `random forest` as models, respectively, by running:


```bash
python code/run_binary_feature_importance_experiments.py --patho lbbb --method shap  --model lr --model rf
```

You can get all rankings for a pathology (e.g., `LBBB`) by simply running:

```bash
python code/run_binary_feature_importance_experiments.py --patho lbbb
```

The rankings are saved as CSV files in `./output`. Here is an overview of the possible command-line arguments:

| Argument | Explanation |
| --- | --- |
| --patho | Defines the pathology (`AVBlock`, `Lbbb`, `Rbbb`) for which the feature importance methods shall be computed. |
| --model | Defines the machine learning model (not relevant if using a filter method). |
| --method | Defines the feature importance method. |

For running the code, use the following abbreviations for the models and methods:

| Model | Abbreviation |
| --- | --- |
| Random Forest | rf |
| Logistic Regression | lr |
| XGB | xgb |
| Neural Network | dn |
| Gaussian Process Classifier | gp |

| Methods | Abbreviation |
| --- | --- |
| SHAP | shap |
| LIME | lime |
| Permutation | permutation |
| Implicit | implicit |

| Filter Methods | Abbreviation |
| --- | --- |
| MRMR | mrmr |
| NCA | nca |
| Relieff | relieff |
| Modified ROCAUC | modifiedrocauc |
| &chi;<sup>2</sup> Test | chisquared |

