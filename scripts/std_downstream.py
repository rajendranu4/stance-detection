from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from transformers import (AutoTokenizer,
                          AutoModelForSequenceClassification, TrainingArguments, Trainer, EarlyStoppingCallback)
import torch
import pandas as pd
import numpy as np
import os
from sklearn.metrics import confusion_matrix

os.environ["WANDB_DISABLED"] = "true"

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
    recall = recall_score(y_true=labels, y_pred=pred)
    precision = precision_score(y_true=labels, y_pred=pred)
    f1 = f1_score(y_true=labels, y_pred=pred)

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


pretrained_model_tokenizer_path = r"C:\Users\rajen\Documents\GitHub\DeCLUTR\output_transformers"
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_tokenizer_path)
model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_tokenizer_path, num_labels=2)

df_input = pd.read_csv(r"C:\Users\rajen\Documents\GitHub\DeCLUTR\path\to\your\dataset\debateforum_bkp.csv")
df_input = df_input.sample(frac=1)

text = list(df_input['text'])
label = list(df_input['label'])

X_train, text_test, y_train, label_test = train_test_split(text, label, test_size=0.2)
text_train, text_val, label_train, label_val = train_test_split(X_train, y_train, test_size=0.2)
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
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=10,
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

model_path = "trained_model"

'''for param in model.roberta.parameters():
    param.requires_grad = False'''

trainer.train()
trainer.save_model(model_path)

# Create torch dataset
text_test_tokenized = tokenizer(text_test, padding=True, truncation=True, max_length=512)
test_dataset = Dataset(text_test_tokenized)

# Load trained model
model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=2)

# Define test trainer
test_trainer = Trainer(model)

# Make prediction
raw_pred, _, _ = test_trainer.predict(test_dataset)

# Preprocess raw predictions
y_pred = np.argmax(raw_pred[0], axis=1)

print("Testing done")
print(y_pred)
print(confusion_matrix(label_test, y_pred))
print(precision_recall_fscore_support(label_test, y_pred))

'''outputs = model(**inputs)
outputs = torch.nn.functional.softmax(outputs.logits, dim=-1)
print(outputs)
print(torch.argmax(outputs, dim=1))'''
