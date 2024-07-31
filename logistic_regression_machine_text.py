import json
import argparse
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from tqdm import tqdm

# Argument parser for command line arguments
parser = argparse.ArgumentParser(description="Train and predict using logistic regression without context.")
parser.add_argument('--train_file', type=str, required=True, help="Path to the training data file (train.json).")
parser.add_argument('--test_file', type=str, required=True, help="Path to the test data file (phase1_test_without_labels.json).")
parser.add_argument('--output_file', type=str, default='answer.json', help="Output file for the predictions (answer.json).")

args = parser.parse_args()

# Load the datasets
train_file_path = args.train_file
test_file_path = args.test_file
output_file_path = args.output_file

train_val_df = pd.read_json(train_file_path)
test_df = pd.read_json(test_file_path)

# Extract sentences and labels
def extract_sentences_and_labels(df):
    sentences = []
    labels = []
    for index, row in df.iterrows():
        sent_and_label = row['sent_and_label']
        sentences.extend([sentence for sentence, _ in sent_and_label])
        labels.extend([1 if label == 'machine' else 0 for _, label in sent_and_label])
    return sentences, labels

# Extract sentences and labels from training data
train_sentences, train_labels = extract_sentences_and_labels(train_val_df)

# Train a logistic regression model
vectorizer = TfidfVectorizer(max_features=3000)
model = LogisticRegression()

pipeline = make_pipeline(vectorizer, model)
pipeline.fit(train_sentences, train_labels)

# Predict labels for the test set
def predict_article_labels(article):
    sentences = article['sent']
    predicted_labels = pipeline.predict(sentences)
    predicted_labels = ['machine' if label == 1 else 'human' for label in predicted_labels]
    answer_format = {"id": article['id'], "sent_and_label": [[sent, pred] for sent, pred in zip(sentences, predicted_labels)], "domain": article['domain']}
    return answer_format

# Generate predictions for each article in the test set
answer_list = []
for index, row in test_df.iterrows():
    answer_format = predict_article_labels(row)
    answer_list.append(answer_format)

# Save the predictions to a JSON file
with open(output_file_path, 'w') as f:
    json.dump(answer_list, f, ensure_ascii=False, indent=4)

print(f"Predictions saved to '{output_file_path}'.")
