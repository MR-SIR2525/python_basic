## Week 7-10 Notes: Advanced Languages

---

### WEEK 7-01: Business Analytics & Data Handling

#### Key Concepts

- **Business Analytics**: Using data, stats, and machine learning to improve decisions.
- **Rule-Based vs Machine Learning**:
  - Rule-Based: Manual, brittle, hard to maintain.
  - ML: Learns from data, adapts to change.

#### Types of Machine Learning

- **Supervised**: Uses labeled data (e.g., classification, regression)
- **Unsupervised**: No labels (e.g., clustering, dimension reduction)

#### Python Tools

- `pandas`: Data handling
- `seaborn`: Visualization (scatterplot, barplot)
- `matplotlib.pyplot`: Plot customization

#### File Reading in Pandas

```python
pd.read_csv('file.csv')
pd.read_excel('file.xlsx')
```

#### Useful Pandas Methods

```python
data.shape     # Get number of rows and columns
data.columns   # Get column names
data.head()    # Preview the first few rows
data['column_name'].fillna(0)
```

---

### WEEK 7-02: Survival Analysis / Logistic Regression

#### Logistic Regression (Classification)

- Logistic regression is a simple and commonly used method to predict the probability of a binary outcome — something that results in either yes or no, 1 or 0 (e.g., good sales vs. bad sales).
- In basic terms: it helps you decide between two categories by estimating the odds of an event happening based on the input data.

##### Connection to Business Statistics:

- If you've taken Business Stats I & II, think of this like a **special case of regression** where the outcome is not a number like revenue or cost, but a **probability** (between 0 and 1).
- It uses the **logit function**, which turns a linear combination of inputs into a probability.
- You still calculate coefficients like in linear regression, but instead of predicting a continuous value, it predicts the chance of something occurring.

##### Details:

- Logistic regression calculates: P(event) = 1 / (1 + e^-z), where z = b0 + b1x1 + b2x2 + ...
- The model draws a decision boundary between classes (usually at 0.5 by default).
- It's useful for problems like: "Will a customer buy or not?", "Is this email spam or not?", etc.

Worked out Example of Logistic Regression

Imagine you're trying to predict whether a product will have **good sales (1)** or **not (0)** based on two features: `discount` and `likes`.

Let's say we train the model and it gives us the equation:



z = -1.5 + 4.0(discount) + 0.00001(likes)

We want to know the probability of good sales for a product with:

- discount = 0.3
- likes = 50,000

**Step 1: Plug into the equation**

z = -1.5 + 4.0(0.3) + 0.00001(50000)\
z = -1.5 + 1.2 + 0.5 = 0.2

**Step 2: Apply the logistic function**

P(good sales) = 1 / (1 + e^-0.2) ≈ 0.55

**Step 3: Interpret it**

- The model estimates there's a 55% chance of good sales.
- If your threshold is 0.5, you classify this as a "good sale".

This is how logistic regression uses your inputs to calculate a probability and make a binary decision.

#### Steps in ML Process

1. Study the problem
2. Choose algorithm
3. Train model
4. Evaluate model

#### Python Code Sketch

```python
model = LogisticRegression()  # Create a logistic regression model object
model.fit(X_train, y_train)   # Train the model using training features and labels
model.predict(X_test)         # Predict outcomes for the test set
```

#### Updating Model Threshold

```python
prob = model.predict_proba(new_input)[0][1]   # Get the probability of the positive class (index 1) for new input
result = 1 if prob > 0.4 else 0               # Classify as 1 (good sale) if probability exceeds 0.4, otherwise 0
```

---

### WEEK 8-01: Support Vector Machine (SVM)

#### SVM Basics

- Finds best hyperplane that separates classes
- Maximizes margin between classes
- Better at minimizing error than logistic regression (in some cases)

#### Model Comparison

- Train both Logistic Regression and SVM
- Compare accuracy, precision, and recall

#### SHAP Values (SHapley Additive exPlanations)

- A tool that helps explain **why** a model made a specific prediction.
- It breaks down a prediction to show how each feature (like discount or likes) helped push the result higher or lower.
- It works kind of like splitting a group project grade — each feature gets credit (or blame) for part of the prediction.
- Bigger SHAP values mean that feature had more impact.
- Helps you understand which inputs mattered most (e.g., maybe likes influenced sales more than discount).

**Visual Example:**

Imagine a model predicts a sales probability of 0.76.

- Base value (average model output across the dataset): 0.50
- SHAP contribution from likes: +0.20
- SHAP contribution from discount: +0.06

**Final prediction:**

- 0.50 + 0.20 + 0.06 = 0.76

This tells us that "likes" contributed more than "discount" to pushing the probability up.

#### Python Code Sketch

```python
model = SVC(kernel='linear', probability=True)
model.fit(X_train, y_train)
```

---

### WEEK 8-02: Decision Trees

#### Decision Tree Basics

- A decision tree is a flowchart-like structure where each internal node represents a decision on a feature (e.g., "Is discount > 0.3?").
- Each branch represents the outcome of that decision, and each leaf node represents a final classification or prediction (e.g., "Good Sale").
- They are easy to interpret and visualize, which makes them useful for explaining decisions to humans.

##### Connection to Binary Trees:

- Decision trees are **similar to binary trees** from Data Structures class:
  - Each node can have **two branches (binary split)**, often based on yes/no conditions.
  - Unlike strict binary trees, some implementations (like CART) force binary splits, while others (like ID3) may allow more.
- Traversing a decision tree is like **walking a binary tree**, making decisions at each node until reaching a leaf.
- You can think of the tree as a recursive structure where each subtree solves a smaller version of the classification problem.

#### Model Comparison Strategy

1. Logistic Regression
2. SVM
3. Decision Tree
4. Use SHAP to explain models

#### Python Code Sketch

```python
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
model.fit(X_train, y_train)
```

---

### WEEK 10: Unsupervised Learning

#### K-Means Clustering

K-Means is a way to group similar data points together without using labels — it's called **unsupervised learning**. You don't tell the algorithm which data points belong together; **it figures that out on its own.**

#### Simple Explanation

- Imagine you have a list of customers with how much they spend and how often they visit your store.
- K-Means will group them into categories like "Big Spenders", "Occasional Visitors", and so on — all based on patterns in the numbers.

#### What is "K"?

- **K** is the number of groups (clusters) you want to find.
- You choose this number ahead of time. For example, K=3 means you want to divide the data into 3 groups.

#### How It Works (Simplified)

1. It starts with K random points.
2. It groups every data point to the nearest one.
3. Then it moves the center of each group.
4. Repeats grouping and moving until nothing changes.

#### Technical Note

- `.values` converts a labeled table (DataFrame) into just numbers so the model can process it.

#### Finding Optimal K

- **Elbow method**: Plot the model error for different K values and look for a sharp "elbow" in the graph where adding more clusters doesn’t help much.
- **Silhouette score**: Measures how well each point fits into its assigned cluster. Higher score = better clustering.

#### Visualization Example

```python
plt.plot(k_values, inertia, marker='o')
plt.title('Elbow-Method')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()
```

#### Applying KMeans

```python
model = KMeans(n_clusters=3)
labels = model.fit_predict(data.values)
```

---

### General Tips

- Always preprocess: handle missing values, scale if needed
- Use SHAP for model explanation
- Check performance using test data (accuracy, confusion matrix)
- Print `data.columns` if you're not sure of field names

