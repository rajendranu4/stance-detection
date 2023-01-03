# -*- coding: utf-8 -*-
"""experiment_model.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Ra-cgXJo9GguBKEzQ5zbJiprVGeJ7Zgd
"""

'''from google.colab import drive
drive.mount('/content/drive')

!pip install transformers'''

import torch
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from transformers import (AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer)
import pandas as pd
import numpy as np
import os
from sklearn.metrics import confusion_matrix



os.environ["WANDB_DISABLED"] = "true"

no_train_epochs = 4
#pretrained_model_tokenizer_path = r"/content/drive/MyDrive/Thesis/output_transformers_e6"
pretrained_model_tokenizer_path = r"/home/prashanth/Documents/Udhay/DeCLUTR/model_dr_df_at_6e"

tokenizer = AutoTokenizer.from_pretrained(pretrained_model_tokenizer_path)

# Create torch dataset
class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels:
            item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.encodings["input_ids"])


def compute_metrics(p):
    pred, labels = p
    pred = np.argmax(pred[0], axis=1)

    print(confusion_matrix(labels, pred))

    accuracy = accuracy_score(y_true=labels, y_pred=pred)
    recall = recall_score(y_true=labels, y_pred=pred, average='macro')
    precision = precision_score(y_true=labels, y_pred=pred, average='macro')
    f1 = f1_score(y_true=labels, y_pred=pred, average='macro')

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_tokenizer_path, num_labels=2)

'''for param in model.roberta.parameters():
    param.requires_grad = False'''

freeze_layer_count = 4
for layer in model.roberta.encoder.layer[:freeze_layer_count]:
    for param in layer.parameters():
        param.requires_grad = False

df_input = pd.read_csv(r"/home/prashanth/Documents/Udhay/DeCLUTR/path/to/your/dataset/DebateForum/Supervised/train/debate_forum_all_topics_train_stratified.csv")
df_input = df_input.sample(frac=1)

text_train = list(df_input['text'])
label_train = list(df_input['label'])

df_input_val = pd.read_csv(r"/home/prashanth/Documents/Udhay/DeCLUTR/path/to/your/dataset/DebateForum/Supervised/val/debate_forum_all_topics_val_stratified.csv")
df_input_val = df_input_val.sample(frac=1)

text_val = list(df_input_val['text'])
label_val = list(df_input_val['label'])

#text_train, text_val, label_train, label_val = train_test_split(text, label, test_size=0.2)
text_train_tokenized = tokenizer(text_train, padding=True, truncation=True, max_length=512)
text_val_tokenized = tokenizer(text_val, padding=True, truncation=True, max_length=512)

train_dataset = Dataset(text_train_tokenized, label_train)
val_dataset = Dataset(text_val_tokenized, label_val)

# Define Trainer
args = TrainingArguments(
    output_dir="output",
    evaluation_strategy="epoch",
    logging_strategy="epoch",
    eval_steps=100,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=no_train_epochs,
    seed=0,

)
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    #callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)

model_path = "/home/prashanth/Documents/Udhay/DeCLUTR/finetuned_models/fine_tuned_model_contrastive_" + str(no_train_epochs) + "_epochs"

trainer.train()
trainer.save_model(model_path)

# Create torch dataset
df_input_val = pd.read_csv(r"/home/prashanth/Documents/Udhay/DeCLUTR/path/to/your/dataset/DebateForum/Supervised/test/debate_forum_all_topics_test_stratified.csv")
df_test = df_test.sample(frac=1)

text_test = list(df_test['text'])
label_test = list(df_test['label'])

text_test_tokenized = tokenizer(text_test, padding=True, truncation=True, max_length=512)
test_dataset = Dataset(text_test_tokenized)

model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=2)

# Define test trainer
test_trainer = Trainer(model)

# Make prediction
raw_pred, _, _ = test_trainer.predict(test_dataset)

# Preprocess raw predictions
y_pred = np.argmax(raw_pred[0], axis=1)

print("Testing done")
print(y_pred)
print("Confusion Matrix:")
print(confusion_matrix(label_test, y_pred))

test_f1 = f1_score(y_true=label_test, y_pred=y_pred, average='macro')
test_accuracy = accuracy_score(y_true=label_test, y_pred=y_pred)
test_recall = recall_score(y_true=label_test, y_pred=y_pred, average='macro')
test_precision = precision_score(y_true=label_test, y_pred=y_pred, average='macro')

print("Test scores")
print("Accuracy: {}\nF1: {}\nPrecision: {}\nRecall: {}\n".format(test_accuracy, test_f1, test_precision, test_recall))
