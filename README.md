# Task 1: Data Handling with NumPy & Pandas

## Objective

To develop a strong foundation in **data loading, cleaning, and manipulation** using **Pandas** and **NumPy** on a real-world student performance dataset.

## Approach

* Loaded the dataset using Pandas and explored its structure using `head()`, `info()`, and `describe()`.
* Checked data quality by verifying **missing values** and **duplicate records**.
* Converted numerical features into NumPy arrays for efficient computation.
* Calculated **mean, median, and standard deviation** using NumPy.
* Applied vectorized operations such as **z-score normalization**.

## Results & Insights

* Dataset contains **10,000 records and 15 features** with no missing or duplicate values.
* Academic and behavioral features show well-distributed statistics.
* NumPy operations enabled fast numerical transformations.
* Normalized data is suitable for further analysis and machine learning tasks.

## Conclusion

This task demonstrates effective use of **Pandas for data handling** and **NumPy for numerical analysis**, preparing the dataset for advanced modeling tasks.


# Task 2: Exploring and Visualizing a Simple Dataset

## Objective
To understand how to read, summarize and visualize a dataset.
## Approach
1. **Import Libraries**  
   Utilized `pandas` for data manipulation, `matplotlib` and `seaborn` for visualization.

2. **Load Dataset**  
   The Iris dataset was loaded from the Seaborn library.

3. **Data Inspection**  
   - Displayed the first few rows (`df.head()`)  
   - Checked the dataset shape (`df.shape`) and columns (`df.columns`)  

4. **Exploratory Data Analysis (EDA)**  
   - **Scatter Plots:** Pairplots to observe relationships between features and species separation.  
   - **Histograms:** Distribution of numeric features by species.  
   - **Box Plots:** Checked for outliers and spread of feature values across species.  

## Results and Insights
- The dataset contains 150 rows and 5 columns (`sepal_length`, `sepal_width`, `petal_length`, `petal_width`, `species`).  
- All columns have no missing values.  
- **Feature importance and relationships:**  
  - Petal length and petal width are the most discriminative features for species classification.  
  - Setosa is easily separable from the other species, while Versicolor and Virginica overlap slightly.  
- **Species characteristics:**  
  - Setosa: smaller petals, shorter and wider sepals.  
  - Versicolor: medium petal and sepal sizes.  
  - Virginica: largest petal dimensions; sepals slightly longer than Setosa.  
- Outliers are present for some features, particularly in sepal width.  

## Conclusion
The Iris dataset is clean, balanced, and ideal for basic exploratory analysis. Visualization techniques like scatter plots, histograms, and box plots reveal the feature distributions and inter-species relationships, helping identify key discriminative features for classification tasks.

