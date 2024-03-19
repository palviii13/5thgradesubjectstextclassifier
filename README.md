# 5th Grade Subjects Text Classifier
This project focuses on text classification using machine learning techniques, specifically Support Vector Machine (SVM) classification. The goal is to categorize text data into predefined subject categories such as Mathematics, Science, History, Geography, and Literature. The project involves data preprocessing, model training, evaluation, and subject prediction for new input content. The goal is to develop a model that can efficiently identify the subject matter of a given text or label it as "irrelevant" if it doesn't pertain to any of these subjects.

# Features
SVM Classification: Utilizes SVM classification for subject categorization.
Data Preprocessing: Cleans and preprocesses text data, including TF-IDF vectorization.
Knowledge Distillation: Explores knowledge distillation techniques to enhance model performance.
Subject Prediction: Enables automated subject prediction for new input content, distinguishing between mathematical expressions and textual descriptions.

# Base Accuracy Measurement
In this project, the first step is training and evaluating a baseline model to establish the foundational accuracies. This involves using traditional machine learning techniques, such as Support Vector Machine (SVM) classification, and basic feature extraction methods like TF-IDF (Term Frequency-Inverse Document Frequency). The base accuracy is 0.967741935483871.

# Fine-Tuning of the Model
During fine-tuning, the text classifier is optimized to enhance accuracy in classifying 5th grade subjects. This involves adjusting hyperparameters such as C and kernel type in SVM classification, employing feature engineering like TF-IDF vectorization and incorporating domain-specific knowledge. These optimizations tailor the model to educational text nuances, improving its precision in subject classification. After fine-tuning, the accuracy was 0.937981981981982

# Datasets
**Used Datasets**

Datasets containing textual content relevant to 5th grade subjects are utilized for training and testing the model. These datasets are sourced from educational materials, textbooks, and curated collections. It is a compilation of data in the form of 3 columns: sr.no, Subject and Content. I could not find any relevant ready to use dataset online so I augmented this dataset by integrating relevant data obtained from a project in a similar domain. The dataset size is 464 rows × 3 columns (4 after including Cleaned_content)

**Preprocessing and Cleaning Dataset**

Before utilizing the dataset for model training, preprocessing and cleaning were performed to ensure data quality and consistency. This involved:

-Removing special characters, punctuation, and numerical digits from the text data.
-Standardizing the text to lowercase to facilitate uniformity.
-Eliminating common stopwords using the NLTK (Natural Language Toolkit) stopwords list.
-These preprocessing steps were applied using a Python function, clean_text(), which was then applied to the dataset's 'Content' column.

**Dataset Analysis**

A comprehensive analysis of the datasets is conducted to gain insights into subject distribution, data imbalance, and potential biases. This analysis guides the selection of appropriate modeling techniques and evaluation strategies.
Subject Distribution: Calculate the distribution of subjects in the dataset to understand the relative frequency of each subject.
Text Length Analysis: Analyze the length of content for each subject by computing statistics such as mean, median, minimum, and maximum text length.
Word Frequency Analysis: Determine the most common words or phrases associated with each subject through word frequency analysis.
TF-IDF Analysis: Compute TF-IDF scores for words across the dataset to identify terms that are most discriminative for each subject.
Visualization: Create visualizations such as bar plots, word clouds, and histograms to visually represent the results of the analyses.

**Test Data Generation**

Train-Test Split

The dataset was split into a training set and a test set with an 80-20 ratio, where 80% of the data was allocated for training (X_train, y_train) and 20% for testing (X_test, y_test). This allocation ensured a sufficient amount of data for training the model while retaining a separate portion for evaluating its performance.

# Distillation
Data Preparation: First,  load and preprocess the text data, including cleaning and vectorizing the text content.

Model Training (Teacher Model): Trained a larger teacher model using the Support Vector Machine (SVM) algorithm with a linear kernel. This model served as the baseline for comparison with accuracy 0.967741935483871. After fine-tuning the accuracy is 0.937981981981982.

Knowledge Distillation: Next step is to generate soft targets (probabilities) from the teacher model's predictions for the training data. Then a smaller student model is trained using these soft targets as labels. Now accuracy is coming to be 0.978494623655914.

Model Evaluation: Size of the teacher model: 111733 bytes
Size of the student model: 111333 bytes
Percentage reduction achieved: 0.35799629473834943 %





