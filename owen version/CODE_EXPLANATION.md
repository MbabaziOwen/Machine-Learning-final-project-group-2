# Code Explanation — Income Prediction Project
### CSC1204 Machine Learning | Uganda Christian University | Easter 2026

This document explains every line of code in `solutions.ipynb`, the theory behind each decision, why the dataset was chosen, and answers to likely exam and viva questions.

---

## Table of Contents

1. [Why This Dataset?](#1-why-this-dataset)
2. [Section A — Problem Definition & Dataset Acquisition](#2-section-a--problem-definition--dataset-acquisition)
3. [Imports & Data Loading](#3-imports--data-loading)
4. [Data Description](#4-data-description)
5. [Missing Values — Detection](#5-missing-values--detection)
6. [Section B — Data Cleaning](#6-section-b--data-cleaning)
7. [Outlier Handling](#7-outlier-handling)
8. [Encoding Categorical Variables](#8-encoding-categorical-variables)
9. [Section B — EDA & Visualisation](#9-section-b--eda--visualisation)
10. [Section C — Model Building](#10-section-c--model-building)
11. [Random Forest](#11-random-forest)
12. [Logistic Regression](#12-logistic-regression)
13. [Model Comparison & Final Conclusion](#13-model-comparison--final-conclusion)
14. [Possible Exam Questions & Answers](#14-possible-exam-questions--answers)

---

## 1. Why This Dataset?

**Dataset chosen:** Adult Census Income Dataset (UCI Machine Learning Repository, 1994 U.S. Census)

### Reasons for choosing it over other datasets

| Criterion | This Dataset | Why it matters |
|---|---|---|
| Size | 32,561 rows | Far above the 200-row minimum — enough data to train models that generalise |
| Features | 14 features + 1 target | Mix of numeric and categorical variables tests full preprocessing pipeline |
| Target variable | Binary (`<=50K` / `>50K`) | Clean binary classification task with a real-world label |
| Real-world relevance | Income inequality, census data | Results are interpretable and socially meaningful |
| Messiness | Missing values encoded as `?` | Requires real data cleaning work, which demonstrates ML competency |
| Known benchmark | Widely cited in ML literature | Results can be validated against published benchmarks |

### Why not other common datasets?

- **Iris / Titanic**: Too simple, too small, overused — would not demonstrate depth.
- **MNIST (image data)**: Requires convolutional networks, not tabular ML.
- **Housing price datasets**: Regression task, not classification.
- **Synthetic datasets**: No real-world backing, less meaningful analysis.

The Adult Census dataset sits in the sweet spot: large enough, messy enough, meaningful enough, and well-suited to binary classification with a mix of feature types.

---

## 2. Section A — Problem Definition & Dataset Acquisition

### The problem statement

The goal is to predict whether a person earns **more or less than $50,000 per year** based on their demographic and employment information.

**This is a supervised binary classification problem** because:
- We have labelled historical data (each row has a known income class)
- The output has exactly two categories (`<=50K` = class 0, `>50K` = class 1)
- We want to learn a mapping from features → label so we can predict for new individuals

**Theory — Supervised Learning:** In supervised learning, a model is trained on input-output pairs `(X, y)`. It learns a function `f(X) ≈ y` by minimising prediction error on the training data. At prediction time, it applies the learned `f` to new inputs.

**Theory — Binary Classification:** A classification problem where the output is one of two discrete categories. Common algorithms include Logistic Regression, Decision Trees, Random Forests, SVMs, and Neural Networks.

---

## 3. Imports & Data Loading

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import warnings
warnings.filterwarnings("ignore")

sns.set_style("whitegrid")

df = pd.read_csv("../adult.csv")
df.head()
```

### Line-by-line explanation

| Line | What it does | Why |
|---|---|---|
| `import pandas as pd` | Loads the pandas library for DataFrames | DataFrames are the standard way to work with tabular data in Python |
| `import numpy as np` | Loads NumPy for numeric operations | Used for array operations, quantile calculations, and heatmap masks |
| `import matplotlib.pyplot as plt` | Core plotting library | Controls figure layout, sizes, titles, and rendering |
| `import seaborn as sns` | Statistical visualisation built on matplotlib | Provides boxplots, heatmaps, countplots, and pairplots with cleaner defaults |
| `from sklearn.model_selection import train_test_split` | Splits data into training and test sets | Prevents the model from being evaluated on data it was trained on |
| `from sklearn.preprocessing import StandardScaler` | Z-score normalisation | Required for Logistic Regression (gradient descent is sensitive to feature scale) |
| `from sklearn.ensemble import RandomForestClassifier` | The primary model | Ensemble of decision trees — robust, handles mixed data types |
| `from sklearn.linear_model import LogisticRegression` | The comparison model | Linear classifier — used to test whether the problem is linearly separable |
| `from sklearn.metrics import ...` | Evaluation functions | Needed to compute accuracy, precision, recall, F1, and full classification reports |
| `warnings.filterwarnings("ignore")` | Suppresses sklearn convergence warnings | Keeps notebook output clean — warnings are expected and not errors |
| `sns.set_style("whitegrid")` | Sets the seaborn plot theme | Adds subtle grid lines to all plots, making values easier to read |
| `df = pd.read_csv("../adult.csv")` | Loads the dataset | `../` goes up one folder level since this notebook is in the `owen version/` subfolder |
| `df.head()` | Displays the first 5 rows | Quick sanity check — confirms the data loaded correctly and reveals column names and types |

**Why `LabelEncoder` was NOT imported:** It was removed because `native.country` is no longer label-encoded — we filter to United States only and drop that column. Importing unused libraries adds clutter.

---

## 4. Data Description

### `df.info()`

```python
df.info()
```

`df.info()` prints the column names, data types, and non-null counts for every column. It serves two purposes:
1. **Confirms data types** — tells us which columns are numeric (`int64`) and which are text (`object`)
2. **Shows apparent completeness** — all 32,561 rows appear non-null, but this is misleading because missing values are stored as the string `"?"`, not `NaN`. pandas cannot see them as missing yet.

**Theory — Data types matter:** Numeric columns can go straight into a model. Categorical (`object`) columns must be encoded into numbers first. Knowing which is which determines the preprocessing steps.

---

### Data Dictionary

```python
descriptions = [
    'Age in years',
    'Type of employer',
    ...
]

data_dict = pd.DataFrame({
    'Type':          df.dtypes,
    'Unique Values': df.nunique(),
    'Description':   descriptions
}, index=df.columns)
```

This builds a custom reference table for every column. Key insight from the output:

- `education` and `education.num` both have **16 unique values** — they encode the same thing (education level), one as text and one as a number. Keeping both would be redundant.
- `fnlwgt` has **21,648 unique values** — it is a census sampling weight, not a personal attribute. It has no predictive value for income.
- `income` has **2 unique values** — confirming binary classification.

---

## 5. Missing Values — Detection

```python
# Missing values are stored as '?' strings, not NaN
missing = df.replace('?', np.nan).isnull().sum()
missing = missing[missing > 0]

for col, count in missing.items():
    print(f'  {col}: {count:,} missing ({count / len(df) * 100:.1f}%)')
print(f'Total: {missing.sum():,} missing entries')
```

### Line-by-line

| Line | What it does | Why |
|---|---|---|
| `df.replace('?', np.nan)` | Temporarily converts `"?"` strings to `NaN` | pandas `.isnull()` only recognises `NaN`/`None`, not arbitrary strings |
| `.isnull().sum()` | Counts true missing values per column | Gives the real missing count after the replacement |
| `missing = missing[missing > 0]` | Filters to only columns that have missing values | Avoids printing 0 for every clean column |
| The `for` loop | Prints the count and percentage for each affected column | Percentage is more informative than raw count — 1.8% vs 5.6% tells you relative severity |

**Result:** Three columns contain missing values:
- `workclass`: 1,836 missing (5.6%)
- `occupation`: 1,843 missing (5.7%)
- `native.country`: 583 missing (1.8%)

**Theory — Why impute instead of drop?** Dropping rows with missing values would remove ~4,262 records (~13% of data). This reduces training data and may introduce bias if missingness is systematic (e.g., unemployed people may not report `workclass`). Imputation preserves all rows.

---

## 6. Section B — Data Cleaning

### Replace `?` with NaN and get descriptive statistics

```python
df = df.replace("?", np.nan)
df.describe().round(2)
```

This time we apply the replacement permanently to `df` (previously it was a temporary copy for counting). `df.describe()` shows count, mean, std, min, quartiles, and max for all numeric columns. Key observations:

- `capital.gain` mean is 1,077 but median (50th percentile) is **0** — extremely right-skewed
- `capital.loss` same pattern — nearly all values are 0
- `hours.per.week` median is exactly **40** — the standard working week dominates

---

### Mode Imputation

```python
for col in ["workclass", "occupation", "native.country"]:
    mode = df[col].mode()[0]
    df[col] = df[col].fillna(mode)
```

| Line | What it does | Why |
|---|---|---|
| `for col in [...]` | Iterates over the three columns with missing values | Avoids writing the same logic three times |
| `df[col].mode()[0]` | Finds the most frequent value in the column | `.mode()` returns a Series (there can be ties); `[0]` takes the first (most common) |
| `df[col].fillna(mode)` | Fills all `NaN` entries in that column with the mode | Replaces missing with the most common value |

**Theory — Why mode imputation for categorical data?**
- Mean/median only apply to numbers. For categories, the "average" is meaningless.
- The **mode** (most frequent value) is the best single-value estimate for a categorical variable.
- Risk: slight over-representation of the dominant category (`Private`, `Prof-specialty`, `United-States`), but at ~5% missingness this is acceptable.

---

### Filter to United States Only

```python
df = df[df["native.country"].str.strip() == "United-States"].copy()
df.drop(columns=["native.country"], inplace=True)
```

| Line | What it does | Why |
|---|---|---|
| `df["native.country"].str.strip()` | Removes leading/trailing whitespace from values | The dataset has whitespace-padded strings like `" United-States"` |
| `== "United-States"` | Boolean mask — True for US residents | Selects only US rows |
| `.copy()` | Creates a fresh independent DataFrame | Prevents `SettingWithCopyWarning` — makes the filtered result a new object, not a view |
| `df.drop(columns=["native.country"], inplace=True)` | Removes the column permanently | After filtering to one country, this column is now constant — it carries zero information |

**Theory — Why drop a constant column?** A feature with only one unique value has zero variance. It contributes nothing to any model's decisions (no split point can be informative). Keeping it wastes memory and computation.

**Why filter to US only?** ~90% of the original data is US residents. Non-US records are few, may have different income dynamics, and create noise. Specialising gives a cleaner, more internally consistent dataset.

---

### Drop Redundant Columns

```python
df.drop(columns=["fnlwgt", "education"], inplace=True)
```

| Column | Why dropped |
|---|---|
| `fnlwgt` | Census sampling weight — tells you how many people this row represents nationally. It is a **survey design variable**, not a personal attribute. Including it would make the model predict census methodology, not income. |
| `education` | Text duplicate of `education.num`. Both encode the same 16 education levels. Keeping both would double-count this feature and cause encoding issues. `education.num` is kept because it is already numeric. |

**Theory — Feature redundancy:** Duplicate features do not add information. In linear models they can cause multicollinearity. In tree models they split importance between two identical signals, making importance scores less interpretable.

---

### Outlier Capping

```python
def cap_outliers(df, col):
    Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    IQR = Q3 - Q1
    upper = df[col].quantile(0.99) if IQR == 0 else Q3 + 1.5 * IQR
    df[col] = df[col].clip(upper=upper)

numeric_cols = ["age", "education.num", "capital.gain", "capital.loss", "hours.per.week"]
fig, axes = plt.subplots(1, 5, figsize=(18, 4))
for ax, col in zip(axes, numeric_cols):
    sns.boxplot(y=df[col], ax=ax, color="lightsteelblue")
    ax.set_title(col, fontsize=10)
plt.suptitle("Boxplots — Before Outlier Capping", fontsize=13)
plt.tight_layout()
plt.show()

cap_outliers(df, "capital.gain")
cap_outliers(df, "capital.loss")
```

---

## 7. Outlier Handling

### The `cap_outliers` function

```python
def cap_outliers(df, col):
    Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    IQR = Q3 - Q1
    upper = df[col].quantile(0.99) if IQR == 0 else Q3 + 1.5 * IQR
    df[col] = df[col].clip(upper=upper)
```

| Line | What it does | Why |
|---|---|---|
| `quantile(0.25)` / `quantile(0.75)` | Computes Q1 (25th percentile) and Q3 (75th percentile) | Standard IQR calculation |
| `IQR = Q3 - Q1` | Interquartile range — the middle 50% spread | Used in the standard outlier fence formula |
| `if IQR == 0` | Special case for zero-inflated columns | When >75% of values are 0, Q1=Q3=0 and IQR=0. The standard formula `Q3 + 1.5 × 0 = 0` would flag ALL non-zero values as outliers, which is wrong. |
| `df[col].quantile(0.99)` | Fallback: use 99th percentile as upper cap | Preserves the meaningful non-zero values while only removing the extreme top 1% |
| `df[col].clip(upper=upper)` | Caps values at the upper bound | Replaces extreme values with the cap rather than deleting rows |

**Theory — Why cap instead of delete?**
- Deleting rows removes training data. Capping preserves the row while reducing the influence of extreme values.
- `clip(upper=upper)` sets any value above the threshold to exactly the threshold value. Values below are untouched.

**Theory — Why only cap `capital.gain` and `capital.loss`?**
- `age` max = 90, min = 17 — plausible human ages, not errors
- `education.num` max = 16 — bounded scale (1–16 education levels), cannot exceed 16
- `hours.per.week` max = 99 — plausible (some people work 90+ hours), not errors
- `capital.gain` max = 99,999 — artificial cap in the original census data collection. The 261 values at exactly 99,999 are censored/truncated, not real. Capping is correct.

### Boxplots before capping

```python
fig, axes = plt.subplots(1, 5, figsize=(18, 4))
for ax, col in zip(axes, numeric_cols):
    sns.boxplot(y=df[col], ax=ax, color="lightsteelblue")
    ax.set_title(col, fontsize=10)
plt.suptitle("Boxplots — Before Outlier Capping", fontsize=13)
plt.tight_layout()
plt.show()
```

| Line | What it does | Why |
|---|---|---|
| `plt.subplots(1, 5, figsize=(18, 4))` | Creates 1 row, 5 column subplot grid | One plot per numeric feature, wide enough to read without overlap |
| `zip(axes, numeric_cols)` | Pairs each axis with a column name | Lets us loop over both simultaneously |
| `sns.boxplot(y=df[col], ax=ax)` | Draws a boxplot on the given axis | `y=` means vertical orientation — the distribution is shown top to bottom |
| `plt.tight_layout()` | Prevents subplots from overlapping | Automatically adjusts spacing |
| `plt.show()` | Renders the figure | Without this, the figure may not display in all notebook environments |

**Theory — What a boxplot shows:**
- The box spans Q1 to Q3 (middle 50% of data)
- The line in the middle is the median
- Whiskers extend 1.5 × IQR beyond the box
- Points beyond the whiskers are outliers

---

## 8. Encoding Categorical Variables

```python
df["income"] = (df["income"].str.strip() == ">50K").astype(int)
df["sex"]    = (df["sex"].str.strip() == "Male").astype(int)

ohe_cols = ["workclass", "marital.status", "occupation", "relationship", "race"]
df = pd.get_dummies(df, columns=ohe_cols, drop_first=True)
```

### Line-by-line

| Line | What it does | Why |
|---|---|---|
| `df["income"].str.strip()` | Removes whitespace from the income string values | The CSV has values like `" >50K"` with a leading space |
| `== ">50K"` | Creates a boolean Series — True for high earners | Comparison produces True/False |
| `.astype(int)` | Converts True→1, False→0 | Models expect numeric labels, not booleans |
| Same pattern for `sex` | `"Male"` → 1, other → 0 | Binary encoding is sufficient for a two-category feature |
| `ohe_cols = [...]` | Lists the nominal categorical columns | These need One-Hot Encoding |
| `pd.get_dummies(df, columns=ohe_cols, drop_first=True)` | One-Hot Encoding with dummy variable drop | Converts each category into a binary column; `drop_first=True` removes the dummy variable trap |

**Theory — Why One-Hot Encoding (OHE)?**
Most ML algorithms work with numbers. A column like `workclass` containing `"Private"`, `"Government"`, `"Self-emp"` cannot be converted to 1, 2, 3 (ordinal encoding) because that implies an order that does not exist. OHE creates one binary column per category: `workclass_Government`, `workclass_Self-emp`, etc.

**Theory — Why `drop_first=True`?**
If you have `k` categories and keep all `k` dummy columns, you introduce perfect multicollinearity — any one column can be perfectly predicted from the others (`Private` = 1 − all other dummies). Dropping one reference category avoids this (the "dummy variable trap"). The dropped category becomes the baseline interpretation.

**Theory — Why not OHE `sex`?**
`sex` has only 2 categories. OHE with `drop_first=True` would produce exactly 1 binary column — identical to our direct binary encoding. The direct approach is simpler.

---

## 9. Section B — EDA & Visualisation

### Target Distribution

```python
counts = df["income"].value_counts().sort_index()
income_label = df["income"].map({0: "<=50K", 1: ">50K"})

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sns.countplot(x=income_label, ax=axes[0], palette=["#4A90D9", "#E8703A"])
...
axes[1].pie(counts, labels=["<=50K", ">50K"], autopct="%1.1f%%", ...)
```

| Line | What it does | Why |
|---|---|---|
| `.value_counts().sort_index()` | Counts 0s and 1s, sorted by index | `.sort_index()` ensures class 0 (<=50K) always comes first for consistent chart labelling |
| `.map({0: "<=50K", 1: ">50K"})` | Converts numeric labels back to readable strings | Chart labels should be human-readable, not just 0/1 |
| `sns.countplot` | Bar chart of class frequencies | Shows the absolute count difference between classes |
| `axes[1].pie(...)` | Pie chart of class proportions | Shows the percentage split — easier to read the imbalance ratio |
| `autopct="%1.1f%%"` | Displays percentage on each pie slice | One decimal place is sufficient precision |

**Key finding:** 75.9% vs 24.1% split — a roughly 3:1 class imbalance. A model that always predicts "<=50K" without learning anything would still score 75.9% accuracy. This is why **F1-score** is the primary evaluation metric.

---

### Histograms and Boxplots by Class

```python
features = ['age', 'education.num', 'hours.per.week', 'capital.gain', 'capital.loss']
fig, axes = plt.subplots(2, 5, figsize=(20, 8))

for i, feat in enumerate(features):
    axes[0, i].hist(df[feat], bins=30, ...)       # full distribution
    axes[0, i].axvline(df[feat].mean(), ...)       # mean line

    low  = df.loc[df['income'] == 0, feat]
    high = df.loc[df['income'] == 1, feat]
    bp = axes[1, i].boxplot([low, high], ...)      # split by class
```

| Line | What it does | Why |
|---|---|---|
| `plt.subplots(2, 5)` | 2 rows × 5 columns — 10 subplots | Top row: overall distribution; bottom row: distribution split by income class |
| `enumerate(features)` | Gets both index `i` and column name | `i` is needed for two-dimensional axis indexing `axes[0, i]` |
| `bins=30` | 30 histogram bins | Enough granularity to see the shape without too much noise |
| `axvline(df[feat].mean())` | Vertical red dashed line at the mean | Visually shows skew when mean ≠ median |
| `df.loc[df['income'] == 0, feat]` | Filters to <=50K values for this feature | Needed to draw separate boxplots per income class |
| `patch_artist=True` | Allows colouring the box fill | Without this, the box interiors cannot be coloured |
| `bp['boxes'][0].set_facecolor(...)` | Colours each class's box differently | Blue = <=50K, Orange = >50K — consistent colour scheme throughout |

---

### Correlation Heatmap

```python
core_cols = ["age", "education.num", "capital.gain", "capital.loss",
             "hours.per.week", "sex", "income"]

corr = df[core_cols].corr()

plt.figure(figsize=(9, 7))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
            mask=np.triu(np.ones_like(corr, dtype=bool)), linewidths=0.5)
```

| Line | What it does | Why |
|---|---|---|
| `core_cols` | Selects only the numeric features | OHE produces 30+ binary columns — including all would make the heatmap unreadable |
| `df[core_cols].corr()` | Pearson correlation matrix | Values range from -1 (perfect negative) to +1 (perfect positive). 0 = no linear relationship |
| `annot=True` | Prints the correlation value in each cell | Without this, you can only read the colour, not the exact value |
| `fmt=".2f"` | Two decimal places for annotations | Consistent precision |
| `cmap="coolwarm"` | Blue-to-red colour scale | Blue = negative correlation, red = positive, white = near zero |
| `mask=np.triu(...)` | Hides the upper triangle | The matrix is symmetric — showing both triangles is redundant |

**Why `native.country` was removed from `core_cols`:** After filtering to United States only and dropping that column, it no longer exists in `df`. Referencing it would cause a `KeyError`.

**Theory — Pearson Correlation:** Measures the linear relationship between two variables. Values close to ±1 indicate strong linear relationships. Does NOT capture non-linear patterns — a tree model may still find value in a feature with low Pearson correlation.

---

### Scatter Plots

```python
sns.scatterplot(data=df, x='age', y='hours.per.week', hue=income_label,
                alpha=0.3, s=8, ax=axes[0], palette=['#4A90D9', '#E8703A'])
```

| Parameter | What it does | Why |
|---|---|---|
| `hue=income_label` | Colours points by income class | Shows whether the two classes cluster in different regions of the feature space |
| `alpha=0.3` | 30% opacity | With ~29,000 points, full opacity would create a solid blob. Transparency reveals density |
| `s=8` | Small point size | Same reason — large points overlap too much at this data volume |

---

### Pairplot

```python
cols   = ['age', 'education.num', 'hours.per.week', 'capital.gain', 'income']
sample = df[cols].sample(2000, random_state=42).copy()
sample['income'] = sample['income'].map({0: '<=50K', 1: '>50K'})

sns.pairplot(sample, hue='income', plot_kws={'alpha': 0.3, 's': 10}, ...)
```

| Line | What it does | Why |
|---|---|---|
| `.sample(2000, random_state=42)` | Takes a random 2,000-row subset | A pairplot on 29,000+ rows would take minutes to render and be unreadable. 2,000 is enough to show patterns. |
| `random_state=42` | Reproducible random seed | Ensures the same 2,000 rows are selected every time the cell is run |
| `.copy()` | Avoids modifying the original `df` | We remap `income` back to strings for labels — we don't want that to affect the modelling data |
| `sns.pairplot` | Creates a grid of scatter plots + diagonal histograms | Shows every pairwise combination of the selected features at once, coloured by class |

**Theory — What a pairplot shows:** Each off-diagonal cell is a scatter plot of two features coloured by class. The diagonal cells show the distribution of each feature per class. Good class separation in any cell means that pair of features is informative.

---

## 10. Section C — Model Building

### Data Preparation

```python
X = df.drop(columns=["income"])
y = df["income"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)
```

| Line | What it does | Why |
|---|---|---|
| `df.drop(columns=["income"])` | Separates features from the target | `X` contains only the inputs; `y` contains only the labels |
| `test_size=0.2` | 80% training, 20% test split | Industry standard. Enough test data to get reliable metric estimates without wasting training data. |
| `random_state=42` | Fixed random seed | Makes the split reproducible — same split every run |
| `stratify=y` | Preserves class ratio in both splits | Without this, random chance could put most >50K records in one split. Stratification guarantees the 75.9%/24.1% ratio is maintained in both train and test. |
| `scaler.fit_transform(X_train)` | Computes mean/std from training data, then scales | Always fit on training data only — fitting on the full dataset would leak test statistics into training (data leakage) |
| `scaler.transform(X_test)` | Scales test data using the **training mean/std** | Applies the same transformation consistently — the test set must be scaled the same way the model was trained |

**Theory — Train/Test Split:** The model must be evaluated on data it has never seen. If you train and evaluate on the same data, you measure memorisation, not generalisation. The test set simulates "unseen real-world data."

**Theory — Why scale for Logistic Regression but not Random Forest?**
- **Logistic Regression** uses gradient descent to find the optimal weights. If `capital.gain` ranges 0–15,024 and `education.num` ranges 1–16, the gradient for `capital.gain` will be orders of magnitude larger, causing the optimizer to take large steps in that direction and tiny steps for other features. StandardScaler normalises all features to mean=0, std=1, making gradient descent converge evenly.
- **Random Forest** splits features on thresholds. The decision `"Is capital.gain > 5,000?"` is equivalent to `"Is scaled_capital.gain > some_threshold?"` — the tree finds the same split either way. Scaling has no mathematical effect on tree decisions.

### `train_eval` helper function

```python
def train_eval(model, X_tr, X_te, y_tr, y_te, label):
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    return {
        "Configuration": label,
        "Accuracy":  round(accuracy_score (y_te, y_pred), 4),
        "Precision": round(precision_score(y_te, y_pred, zero_division=0), 4),
        "Recall":    round(recall_score   (y_te, y_pred, zero_division=0), 4),
        "F1-Score":  round(f1_score       (y_te, y_pred, zero_division=0), 4),
    }
```

| Line | What it does | Why |
|---|---|---|
| `model.fit(X_tr, y_tr)` | Trains the model on training data | The model learns the mapping from features to labels |
| `model.predict(X_te)` | Generates predictions on unseen test data | The model applies what it learned to new inputs |
| `zero_division=0` | Returns 0 instead of a warning when a class has no predictions | Prevents errors with heavily imbalanced configurations that predict one class only |
| `round(..., 4)` | Four decimal places | Consistent precision across all metrics for easy comparison |

**Theory — Evaluation Metrics:**
- **Accuracy** = (correct predictions) / (total predictions). Misleading with imbalanced classes.
- **Precision** = TP / (TP + FP). Of all predicted >50K, how many actually were >50K? Measures false alarm rate.
- **Recall** = TP / (TP + FN). Of all actual >50K earners, how many did the model catch? Measures miss rate.
- **F1-Score** = 2 × (Precision × Recall) / (Precision + Recall). Harmonic mean — penalises extremes. The primary metric when classes are imbalanced.

---

## 11. Random Forest

### Algorithm Justification

Random Forest was chosen as the **primary model** because:
1. It is an ensemble — it averages predictions from many trees, reducing overfitting (variance reduction via bagging)
2. It handles the mix of numeric and binary OHE features without additional preprocessing
3. It provides **feature importances** — directly answers the project objective of "which features predict income?"
4. It is robust to outliers since it makes split decisions on rankings, not absolute values
5. It typically outperforms single decision trees and is competitive with more complex models on tabular data

### Configuration Code

```python
rf_configs = [
    ("Config 1 — Default (100 trees, no depth limit)",
     RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)),

    ("Config 2 — 200 trees, max_depth=10",
     RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)),

    ("Config 3 — 100 trees, max_depth=5",
     RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_split=10,
                            random_state=42, n_jobs=-1)),
]

rf_results = [train_eval(m, X_train, X_test, y_train, y_test, l) for l, m in rf_configs]
rf_results_df = pd.DataFrame(rf_results)

best_idx = rf_results_df["F1-Score"].idxmax()
best_rf_label, best_rf_model = rf_configs[best_idx]
rf_results_df
```

### Hyperparameter explanations

| Hyperparameter | Config 1 | Config 2 | Config 3 | Effect |
|---|---|---|---|---|
| `n_estimators` | 100 | 200 | 100 | More trees = lower variance, slower training |
| `max_depth` | None (unlimited) | 10 | 5 | Limits how deep each tree grows. Shallower = less overfitting but may miss patterns |
| `min_samples_split` | 2 (default) | 2 (default) | 10 | Minimum samples needed to split a node. Higher = less overfitting |
| `n_jobs=-1` | All three | All three | All three | Use all CPU cores for parallel tree training |
| `random_state=42` | All three | All three | All three | Reproducible results |

**Why three configs?** To demonstrate the precision-recall tradeoff as the model is regularised:
- Config 1 (unlimited depth) → highest recall (captures more true positives)
- Config 2/3 (limited depth) → higher precision but lower recall (more conservative)

**List comprehension instead of a for loop:**
```python
rf_results = [train_eval(m, X_train, X_test, y_train, y_test, l) for l, m in rf_configs]
```
This is equivalent to a for loop but more concise. It iterates over `rf_configs` (a list of `(label, model)` tuples), unpacks each tuple into `l` and `m`, calls `train_eval`, and collects all results into a list.

**Selecting the best model:**
```python
best_idx = rf_results_df["F1-Score"].idxmax()
best_rf_label, best_rf_model = rf_configs[best_idx]
```
`.idxmax()` returns the **index** (row number) of the maximum F1-Score. We use that index to retrieve the corresponding label and model object from `rf_configs`.

### Best model report

```python
y_pred_rf = best_rf_model.predict(X_test)
print(classification_report(y_test, y_pred_rf, target_names=["<=50K", ">50K"]))
```

`classification_report` prints precision, recall, F1-score, and support (number of samples) for each class. The best RF configuration is Config 1 (unlimited depth, 100 trees), selected because it achieves the highest F1-score on the minority class (>50K).

### Feature Importances

```python
importances = pd.Series(best_rf_model.feature_importances_, index=X.columns)
top15 = importances.nlargest(15).sort_values()

top15.plot(kind="barh", ...)
```

| Line | What it does | Why |
|---|---|---|
| `best_rf_model.feature_importances_` | Array of importance scores, one per feature | Sklearn computes this as the mean decrease in impurity (Gini impurity) across all trees and splits |
| `pd.Series(..., index=X.columns)` | Attaches column names to the scores | Without this, the array has no labels |
| `.nlargest(15)` | Selects the top 15 most important features | 15 is enough to show the meaningful features without cluttering the chart |
| `.sort_values()` | Sorts ascending (smallest at top) | For a horizontal bar chart, ascending order puts the most important feature at the top |
| `kind="barh"` | Horizontal bar chart | Feature names are long — horizontal bars give more space for labels |

**Theory — Feature Importance (Mean Decrease in Impurity):** Each time a feature is used to split a node in a tree, the impurity (Gini or entropy) decreases. Summing these decreases across all trees and normalising to sum to 1 gives the importance score. A feature used frequently at high-level splits (affecting many samples) gets a higher score.

---

## 12. Logistic Regression

### Algorithm Justification

Logistic Regression was chosen as the **comparison model** because:
1. It is the simplest and most interpretable classifier for binary classification
2. It provides a meaningful contrast to Random Forest — linear vs. non-linear
3. If LR performs comparably to RF, it tells us the class boundary is largely linear
4. Coefficients are interpretable as log-odds — each feature's direction and magnitude of effect can be read directly

### Configuration Code

```python
lr_configs = [
    ("Config 1 — C=1.0 (default)",
     LogisticRegression(C=1.0, max_iter=1000, random_state=42)),

    ("Config 2 — C=0.01 (stronger regularisation)",
     LogisticRegression(C=0.01, max_iter=1000, random_state=42)),
]
```

| Hyperparameter | Effect |
|---|---|
| `C` | Inverse of regularisation strength. **High C** = less regularisation (model fits training data more closely). **Low C** = more regularisation (coefficients shrink toward zero, simpler model). |
| `max_iter=1000` | Maximum iterations for the gradient descent solver. Default (100) is often insufficient for this dataset size — increasing prevents a `ConvergenceWarning`. |

**Why only two configs for LR?** LR has fewer hyperparameters than RF. The primary lever is `C`. Two configs demonstrate the regularisation tradeoff without over-engineering.

**Theory — Logistic Regression:** Models the probability `P(y=1 | X)` as a sigmoid function of a linear combination of features:
```
P(y=1) = 1 / (1 + e^(-(w₀ + w₁x₁ + w₂x₂ + ...)))
```
The model predicts class 1 if `P(y=1) > 0.5`. Training finds the weights `w` that minimise the log-loss (cross-entropy) between predicted probabilities and actual labels. L2 regularisation adds a penalty `λ||w||²` to prevent any weight from growing too large.

---

## 13. Model Comparison & Final Conclusion

| Metric | Random Forest | Logistic Regression | Winner |
|---|---|---|---|
| Accuracy | 0.8454 | **0.8511** | LR |
| Precision (>50K) | 0.71 | **0.73** | LR |
| Recall (>50K) | **0.61** | 0.60 | RF |
| **F1-Score (>50K)** | 0.6543 | **0.6613** | **LR** |

**Final winner: Logistic Regression (Config 1)**

The fact that a simple linear model slightly outperforms a complex ensemble suggests the **income class boundary is largely linear** in this encoded feature space. The dominant features (education level, age, capital gain, marital status) relate to income in a relatively monotonic, linear fashion once properly encoded.

**Both models struggle with recall on the minority class.** This is a direct consequence of the 3:1 class imbalance. The model never sees enough >50K training examples to learn all the patterns in that class. Future improvements:
- `class_weight='balanced'` — tells the model to penalise errors on the minority class more heavily
- SMOTE (Synthetic Minority Oversampling Technique) — generates synthetic >50K training examples
- Lower prediction threshold — instead of `P(>50K) > 0.5`, use `> 0.3` to increase recall at the cost of precision

---

## 14. Possible Exam Questions & Answers

### On the Dataset

**Q: Why was the Adult Census Income dataset chosen for this project?**
A: It meets all project requirements (32,561 rows, 14 features, mix of numeric and categorical types, clear binary target). It is also a well-known benchmark dataset with messy real-world characteristics (missing values encoded as `"?"`, skewed features, class imbalance) that require a complete preprocessing pipeline, demonstrating ML competency.

**Q: Why was the analysis restricted to United States residents only?**
A: ~90% of records are from the United States. Non-US records are too few to represent their respective countries fairly and may follow different income dynamics (different economies, currencies, norms). Filtering produces a more internally consistent dataset.

**Q: Could other datasets have been used? Give an example.**
A: Yes. The Titanic dataset (survival prediction), Bank Marketing dataset (loan subscription prediction), or Credit Card Fraud dataset (fraud detection) are all suitable binary classification datasets. The Adult dataset was chosen for its size, real-world relevance, and interpretable features.

---

### On Missing Values

**Q: Why were missing values stored as `"?"` rather than `NaN`?**
A: This is an artifact of how the original 1994 census data was encoded. Some data collection systems store blanks or unknowns as placeholder characters rather than null values. pandas reads these as regular strings, so explicit detection and replacement is required.

**Q: Why was mode imputation used instead of dropping rows?**
A: Dropping 4,262 rows (~13% of data) would waste training data and could introduce bias if the missingness is not random (e.g., unemployed people systematically not reporting `workclass`). Mode imputation preserves all rows while filling missing values with the most plausible single value for categorical features.

**Q: What is the risk of mode imputation?**
A: It slightly over-represents the dominant category. For example, filling 583 missing `native.country` values with `"United-States"` makes the US appear even more dominant. However, since we then filter to US only, this has no net effect here. For `workclass` and `occupation`, the over-representation effect is small (~5% of rows).

---

### On Outliers

**Q: Why was `capital.gain` capped at the 99th percentile instead of using IQR?**
A: Over 75% of values are exactly 0, making Q1 = Q3 = 0 and IQR = 0. The standard fence `Q3 + 1.5 × IQR = 0` would classify every non-zero value as an outlier, destroying a highly informative feature. The 99th percentile correctly handles this zero-inflated distribution by only removing the extreme top 1%.

**Q: Why were `age`, `education.num`, and `hours.per.week` not capped?**
A: These features have natural, plausible upper bounds. An age of 90, a 16-level education, or 99 hours/week are extreme but believable real-world values, not data errors. Capping them would destroy valid information.

---

### On Encoding

**Q: What is the dummy variable trap and how was it avoided?**
A: When you one-hot encode a variable with `k` categories, the `k` dummy columns are perfectly multicollinear (they sum to 1 for every row). Including all `k` causes mathematical instability in linear models. Using `drop_first=True` in `pd.get_dummies` removes one reference category, breaking the multicollinearity.

**Q: Why was `education.num` kept as a numeric feature rather than one-hot encoded?**
A: `education.num` is an ordinal scale from 1 to 16, where higher numbers genuinely represent more education. Preserving the numeric encoding respects this ordering and keeps it as a single feature. OHE would expand it into 15 binary columns, losing the ordinal structure and increasing dimensionality unnecessarily.

**Q: What is the difference between ordinal encoding and one-hot encoding?**
A: Ordinal encoding assigns integers (1, 2, 3...) to categories, implying an order. OHE creates a binary column per category with no implied order. Use ordinal encoding when an order exists (education levels, ratings). Use OHE when categories are nominal (workclass, race, occupation).

---

### On Train/Test Split

**Q: What is the purpose of `stratify=y` in `train_test_split`?**
A: It ensures the class ratio (75.9% / 24.1%) is preserved in both the training and test sets. Without stratification, random chance could result in the test set having a different ratio, making evaluation metrics unrepresentative of real-world performance.

**Q: Why fit the scaler on training data only, not the full dataset?**
A: Fitting on the full dataset would use information from the test set to compute the mean and standard deviation. This is called **data leakage** — the model indirectly "sees" test data during training. To simulate a truly unseen evaluation, all transformations must be learned from training data only, then applied to test data.

**Q: Why use an 80/20 split and not 70/30 or 90/10?**
A: 80/20 is the industry convention for this dataset size. With ~29,000 rows, 20% (≈5,800 samples) provides statistically reliable metric estimates. A 90/10 split gives fewer test samples (less reliable evaluation); 70/30 wastes more training data. The exact split matters less at large dataset sizes.

---

### On Random Forest

**Q: What is bagging and how does Random Forest use it?**
A: Bagging (Bootstrap Aggregating) trains each tree on a **random bootstrap sample** of the training data (sampling with replacement). This means each tree sees a slightly different dataset and makes different errors. Averaging their predictions reduces variance — the ensemble is more stable and generalises better than any single tree.

**Q: What is the difference between Random Forest and a single Decision Tree?**
A: A single decision tree fits training data exactly (high variance, prone to overfitting). Random Forest builds many trees, each on a random data sample and random feature subset, then averages their predictions. The averaging cancels out individual errors, reducing overfitting. RF consistently outperforms single trees on real-world data.

**Q: What does `n_jobs=-1` do?**
A: It tells scikit-learn to use all available CPU cores for training. Each tree can be trained independently in parallel, so more cores = faster training. With 100 trees, this can reduce training time by a factor equal to the number of cores.

**Q: Why did Config 1 (unlimited depth) outperform Configs 2 and 3?**
A: Unlimited depth allows trees to learn finer patterns. While individual trees may overfit, the ensemble averaging in a 100-tree forest is sufficient regularisation for this dataset. Configs 2 and 3 over-regularise — the depth limit causes underfitting (high bias), losing more recall than they gain in precision.

---

### On Logistic Regression

**Q: What does the `C` parameter control in Logistic Regression?**
A: `C` is the inverse of regularisation strength. `C=1.0` is the default (moderate regularisation). `C=0.01` is strong regularisation — coefficients are heavily penalised for being large, pushing the model toward a simpler boundary. `C=100` would be weak regularisation, allowing the model to fit training data more closely (more risk of overfitting).

**Q: Why does Logistic Regression require feature scaling but Random Forest does not?**
A: LR uses gradient descent — the update step for each weight is proportional to the feature's value. Large-scale features dominate the gradient updates, causing slow or uneven convergence. RF makes binary split decisions (`feature > threshold?`) — these are invariant to the scale of the feature. Multiplying all values by 100 moves the threshold by 100 but finds the same split.

**Q: What is L2 regularisation?**
A: L2 regularisation adds a penalty term `λ × Σwᵢ²` to the loss function. This penalises large weights, encouraging the model to spread importance across many small weights rather than relying heavily on one feature. The result is a smoother decision boundary that generalises better. In sklearn's LR, `C = 1/λ`.

---

### On Model Comparison

**Q: Which model performed better and why?**
A: Logistic Regression (Config 1) achieved a slightly higher F1-Score (0.6613 vs 0.6543). This suggests the income class boundary is largely linear in the encoded feature space. The dominant features — education level, age, capital gain — all have relatively monotonic relationships with income that a linear model captures well.

**Q: What does it mean when a linear model outperforms a complex ensemble?**
A: It suggests the decision boundary between classes is approximately linear (or that the non-linear patterns are weak enough that the ensemble's added complexity does not help enough to justify it). The feature transformations (OHE, binary encoding) have already captured most of the non-linearity.

**Q: Both models only recall ~60% of >50K earners. How could this be improved?**
A: Several strategies:
1. **`class_weight='balanced'`** — automatically adjusts the loss function to penalise minority class errors more heavily
2. **SMOTE** — generates synthetic minority class training examples to balance the training set
3. **Lower decision threshold** — predict >50K when `P(>50K) > 0.3` instead of `> 0.5`, increasing recall at the cost of precision
4. **Collect more >50K data** — the fundamental fix for imbalance

**Q: Why is F1-score the primary metric instead of accuracy?**
A: With a 3:1 class imbalance, a model that always predicts `<=50K` achieves 75.9% accuracy while being completely useless for identifying high earners. F1-score is the harmonic mean of precision and recall — it cannot be high unless the model identifies a substantial proportion of true positives without too many false positives. It is the most balanced metric for imbalanced classification.

**Q: What is the difference between precision and recall? When would you prioritise each?**
A:
- **Precision** = of all predicted positives, how many were actually positive. Prioritise when false positives are costly (e.g., flagging innocent people as fraudsters).
- **Recall** = of all actual positives, how many were detected. Prioritise when false negatives are costly (e.g., missing a cancer diagnosis, missing a high earner who should receive a service).
- For income prediction, missing a >50K earner (false negative) may be more costly than incorrectly identifying a <=50K earner as >50K, so recall is often emphasised.

---

*This document covers the complete `solutions.ipynb` notebook — every code decision, its theoretical justification, and questions a lecturer or examiner is likely to ask.*
