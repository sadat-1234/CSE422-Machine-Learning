# Software Quality Classification

An academic machine learning project that explores a software quality dataset, predicts **High**, **Low**, and **Medium** quality labels, and compares supervised classifiers. It also uses K-Means clustering and PCA to explore patterns in the feature space.

Developed for **CSE422**, Section 10, Group 12.

## Features

- Exploratory analysis of numerical and categorical features.
- Missing-value inspection, distributions, box plots, and correlation visualizations.
- Categorical encoding and numerical standardization.
- Stratified training/test splitting.
- Comparison of Logistic Regression, Gaussian Naive Bayes, and a multilayer perceptron.
- K-Means clustering with three clusters.
- PCA visualization of the clustering results.
- Accuracy, weighted precision, weighted recall, confusion matrices, and one-vs-rest ROC curves.

## Technologies

Python, pandas, NumPy, Matplotlib, Seaborn, scikit-learn, and Jupyter Notebook.

## Included file

| File | Purpose |
| --- | --- |
| `Final_CSE422_Project.ipynb` | Dataset analysis, preprocessing, classification, clustering, and evaluation |

## Dataset requirement

The notebook reads:

```python
main_df = pd.read_csv('software_quality_dataset.csv')
```

The CSV is **not included** in the uploaded project. Place the original `software_quality_dataset.csv` beside the notebook before running it. A substitute CSV must match the columns and categories referenced by the notebook, including `Quality_Label` and the expected encoded feature names.

The supervised target is reconstructed as:

| Label | Code |
| --- | --- |
| High | 0 |
| Low | 1 |
| Medium | 2 |

The notebook contains some older comments referring to delivery-related features. The loaded filename and classification target identify the implemented task as software quality classification.

## Model configuration

| Component | Configuration shown in the notebook |
| --- | --- |
| Logistic Regression | scikit-learn defaults |
| Gaussian Naive Bayes | scikit-learn defaults |
| Multilayer perceptron | Two hidden layers of 50 units; `max_iter=500`; `random_state=42` |
| K-Means | Three clusters; `random_state=42` |
| PCA | Two components for visualization |
| Train/test split | 70% / 30%, stratified by target, `random_state=42` |

Other classifiers appear in the imports, but the final supervised comparison uses the three classifiers listed above.

## Getting started

1. Download the notebook and place the original CSV in the same directory.
2. Install the dependencies:

   ```bash
   python -m pip install jupyter numpy pandas matplotlib seaborn scikit-learn
   ```

3. Start Jupyter:

   ```bash
   jupyter notebook
   ```

4. Open `Final_CSE422_Project.ipynb`.
5. Before the exploratory analysis cells that reference `eda_df`, add:

   ```python
   eda_df = main_df.copy()
   ```

   The supplied notebook references this variable without an assignment in its saved source.

6. Run the notebook cells in order and inspect the generated tables and figures.

A scikit-learn installation that supports `OneHotEncoder(sparse_output=False)` is required. The project does not provide a pinned dependency environment.

## Evaluation

The notebook compares classification models using accuracy, weighted precision and recall, confusion matrices, and one-vs-rest ROC/AUC. K-Means is an exploratory clustering step; its cluster numbers do not automatically correspond to the three quality labels.

Recorded figures and outputs are retained in the notebook. They should be interpreted as the results of the original academic workflow.

## Methodological limitations

The original notebook fits encoding and standardization before the train/test split. For a rigorous new evaluation, split first and fit preprocessing on the training subset only. Results from the original workflow should not be treated as a leakage-free benchmark.

The dataset is not bundled, and the notebook needs the `eda_df` initialization described above for a fresh run.

## Contributors

- Sahir Jawad Chowdhury
- Ihfaz Rashid Sadat

This was a group academic project; the original notebook records the course and group details.

