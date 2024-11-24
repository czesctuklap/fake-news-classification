# Fake News Classification

The project is focused on classifying fake news articles using machine learning techniques such as `Multilayer Perceptron` and `k-Nearest Neighbors` algorithm. It involves preprocessing the data, feature extraction and building predictive models.

### Data Preprocessing

The data preprocessing pipeline includes the following steps:
- **Text Cleaning**: Removing unwanted characters, special symbols, and stopwords to retain meaningful content.
- **Normalization**: Converting all text to lowercase and applying stemming or lemmatization for consistency.
- **Vectorization**: Transforming text data into numerical features using techniques like `Term Frequency-Inverse Document Frequency`.
  
### Feature Extraction

To enhance the model's understanding of the text data, the `TF-IDF` method is used, representing text in a high-dimensional vector space. The formula for `TF-IDF` is:

![image](https://github.com/user-attachments/assets/0976e10b-6f45-4039-875e-59393c5c2bce)

where:
- **tf(t, d)**: Term frequency of term `t` in document `d`,
- **idf(t, D)**: Inverse document frequency of term `t`, with `N` being the total number of documents and `d ∈ D : t ∈ d` the number of documents where the term `t` appears.

To reduce the complexity of high-dimensional data and improve model efficiency, `Principal Component Analysis` is used. `PCA` reduces the feature space while retaining the most important information by projecting data onto a set of orthogonal axes (principal components). This transformation helps:
- Minimize redundancy in the dataset.
- Speed up training and prediction times.
- Mitigate the risk of overfitting.

The number of principal components is chosen based on the explained variance ratio, ensuring that the reduced feature set captures a significant proportion of the original data's variance.

### Model Training

The project uses the following machine learning models to classify fake news:
- **k-Nearest Neighbors**: A simple algorithm that classifies data points based on the majority class of their nearest neighbors. The distance between points is calculated using metrics like Euclidean distance, and the optimal value of `k` is determined through cross-validation.
- **Multilayer Perceptron**: A feedforward neural network that uses backpropagation to adjust weights and biases for optimal performance. The `MLP` model is designed to learn complex patterns in the data through its hidden layers and non-linear activation functions.

The models are evaluated using metrics such as:
- Accuracy
- Precision
- Recall
- F1-score

### Visualization and Insights

The project provides:
- A summary of key metrics for each model,
- Database Analysis,
- Visual Division of Real and Fake News,
- Visualisation of Most Frequent words in Titles and Text,
- Graph of division of extracted by `PCA` and `TF-IDF` features by labels,
- Visualisation of Most Frequent words in Fake News and Real Articles,
- Graph of Co-Occurence frequencies of words in Titles,
- Graph of dependencies of `k` and accuracy in `kNN`,
- Plot of the Model's Accuracy and Loss

### Dataset

Dataset: https://www.kaggle.com/c/fake-news/data
