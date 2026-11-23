# Practical 08: Decision Trees and Max Depth Analysis

## Student Information
- **Name:** Ankush Kumar Singh[cite: 2, 5]
- **Registration Number:** GF202459590[cite: 2, 5]
- **Subject:** Statistical Foundation of Data Science[cite: 5]

## Objective
Build and evaluate a Decision Tree classifier on `diabetes_prediction_dataset.csv` (`100,000` rows, `9` columns) to predict `diabetes`, compute target entropy, Information Gain, and Gini reduction across features, visualize the tree and confusion matrix, and compare model accuracy across `max_depth` values from `1` to `10`[cite: 50, 51, 53, 57, 59].

## Files Included
- `Practical8.ipynb.pdf` — Exported Google Colab notebook (`Practical 8.ipynb`) containing preprocessing, feature split metrics, tree visualization, evaluation metrics, and `max_depth` comparison[cite: 50, 53, 57, 58, 59]
- `diabetes_prediction_dataset.csv` — Dataset used for training and testing (uploaded in Colab or placed in this folder)[cite: 50, 51]

## Concepts, Analysis & Methods Covered
- **Dataset Exploration & Stratified Split:** Inspecting `100,000` records (`8` features, binary target `diabetes`: `91.5%` class `0`, `8.5%` class `1`) and splitting into `80%` training (`80,000` rows) and `20%` testing (`20,000` rows) sets using `train_test_split(..., stratify=y, random_state=42)`[cite: 51, 52, 53]
- **Entropy, Information Gain & Gini Reduction:** Computing target entropy (`~0.4196`) and single-split impurity reductions per feature, finding `HbA1c_level` as the top feature (`Information Gain ≈ 0.1523`, `Gini Reduction ≈ 0.0676`)[cite: 53, 54, 55]
- **Preprocessing & Decision Tree (`max_depth=4`):** Using a `ColumnTransformer` (`OneHotEncoder` on `gender` and `smoking_history`; passthrough on numerical features) inside a `Pipeline` with `DecisionTreeClassifier(criterion="entropy", max_depth=4, random_state=42)` (`HbA1c_level` as root feature around `6.7`), achieving `97.23%` test accuracy (`0.9723`)[cite: 56, 57, 58]
- **Model Evaluation & Depth Comparison:** Generating confusion matrix (`[[18300, 0], [554, 1146]]`), classification report, and testing `max_depth` from `1` to `10` (`0.95450` at depth `1`, `0.97230` across depths `2`–`9`, and `0.97195` at depth `10`)[cite: 58, 59]

## How to Run / View
1. View `Practical8.ipynb.pdf` directly on GitHub, or open `Practical 8.ipynb` in Google Colab / Jupyter Notebook[cite: 50].
2. Upload or point to `diabetes_prediction_dataset.csv` when running the notebook[cite: 50, 51].
3. Run all cells sequentially to reproduce the feature rankings, decision tree plot, confusion matrix, classification report, and depth comparison table[cite: 53, 54, 55, 57, 58, 59].
