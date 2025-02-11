import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

# Load and preprocess the data
df = pd.read_csv('spam.csv', sep=',')

# Balance the dataset
spamDF = df[df['label'] == 'spam']
hamDF = df[df['label'] == 'ham']
hamDF = hamDF.sample(spamDF.shape[0])  # Downsample ham to match spam
finalDF = pd.concat([hamDF, spamDF])

# Initialize accuracy lists
train_accuracies_rf = []
test_accuracies_rf = []
train_accuracies_svc = []
test_accuracies_svc = []

# Run the experiment over 100 iterations
for i in range(100):
    # Split the dataset into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(
        finalDF['message'], finalDF['label'], test_size=0.2, random_state=i, stratify=finalDF['label']
    )
    
    # Define TF-IDF vectorizer and classifiers with pipelines
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), max_features=5000)
    model1 = Pipeline([('tfidf', tfidf_vectorizer), ('model', RandomForestClassifier(n_estimators=100, n_jobs=-1))])
    model2 = Pipeline([('tfidf', tfidf_vectorizer), ('model', SVC(C=1000, gamma='auto'))])
    
    # Train the models
    model1.fit(x_train, y_train)
    model2.fit(x_train, y_train)
    
    # Predictions
    y_train_predict1 = model1.predict(x_train)
    y_train_predict2 = model2.predict(x_train)
    y_test_predict1 = model1.predict(x_test)
    y_test_predict2 = model2.predict(x_test)
    
    # Accuracy scores
    train_accuracies_rf.append(accuracy_score(y_train, y_train_predict1))
    test_accuracies_rf.append(accuracy_score(y_test, y_test_predict1))
    train_accuracies_svc.append(accuracy_score(y_train, y_train_predict2))
    test_accuracies_svc.append(accuracy_score(y_test, y_test_predict2))

# Compute average accuracies
avg_train_accuracy_rf = np.mean(train_accuracies_rf)
avg_test_accuracy_rf = np.mean(test_accuracies_rf)
avg_train_accuracy_svc = np.mean(train_accuracies_svc)
avg_test_accuracy_svc = np.mean(test_accuracies_svc)

# Generate bar chart for average accuracies
classifiers = ['Random Forest', 'Support Vector']
training_accuracies = [avg_train_accuracy_rf, avg_train_accuracy_svc]
testing_accuracies = [avg_test_accuracy_rf, avg_test_accuracy_svc]

x = np.arange(len(classifiers))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
bars1 = ax.bar(x - width/2, training_accuracies, width, label='Training Accuracy')
bars2 = ax.bar(x + width/2, testing_accuracies, width, label='Test Accuracy')

# Add labels and title
ax.set_xlabel('Classifiers')
ax.set_ylabel('Accuracy')
ax.set_title('Average Training and Test Accuracy by Classifier (100 Iterations)')
ax.set_xticks(x)
ax.set_xticklabels(classifiers)
ax.legend()

# Annotate bars with values
for bar in bars1 + bars2:
    height = bar.get_height()
    ax.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')

# Save the chart as an image for LaTeX document
plt.tight_layout()
plt.savefig('average_accuracy_comparison.png')
plt.show()

# Print average accuracies for reference
print("Average Training Accuracy (Random Forest):", avg_train_accuracy_rf)
print("Average Test Accuracy (Random Forest):", avg_test_accuracy_rf)
print("Average Training Accuracy (SVC):", avg_train_accuracy_svc)
print("Average Test Accuracy (SVC):", avg_test_accuracy_svc)
