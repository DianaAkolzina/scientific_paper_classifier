import os
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support, confusion_matrix
from transformers import BertTokenizer, BertModel, AdamW
from torch.utils.data import DataLoader, TensorDataset
from torch import nn
from tqdm import tqdm
from google.cloud import bigquery, storage
import numpy as np
import logging
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

client = bigquery.Client()
query = """
SELECT Processed_Text,  Label
FROM data_train.processed_text_10000
"""
df = client.query(query).to_dataframe()

text_column = 'Processed_Text'
label_column = 'Label'

combined_df = df
combined_df.reset_index(drop=True, inplace=True)
combined_df['Label'] = combined_df['Label'].astype('category')

label_counts = combined_df['Label'].value_counts()
min_category_size = label_counts.min()
balanced_df = pd.concat([
    combined_df[combined_df['Label'] == label].sample(n=min_category_size, random_state=42)
    for label in combined_df['Label'].cat.categories
])
balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

X = balanced_df['Processed_Text']
y = balanced_df['Label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

tokenizer = BertTokenizer.from_pretrained('allenai/scibert_scivocab_uncased', do_lower_case=True)
bert_model = BertModel.from_pretrained('allenai/scibert_scivocab_uncased')

class SciBERTClassifier(nn.Module):
    def __init__(self, num_labels):
        super(SciBERTClassifier, self).__init__()
        self.bert = bert_model
        self.dropout = nn.Dropout(0.1)
        self.linear1 = nn.Linear(768, 256)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(256, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask, output_attentions=True)
        pooled_output = outputs[1]
        attention_weights = outputs[2]
        pooled_output = self.dropout(pooled_output)
        linear1_output = self.linear1(pooled_output)
        relu_output = self.relu(linear1_output)
        final_output = self.linear2(relu_output)
        return final_output, attention_weights

num_labels = len(balanced_df['Label'].unique())
model = SciBERTClassifier(num_labels)
model.to(device)

max_length = 256
batch_size = 32

def tokenize_data(texts, labels):
    input_ids = []
    attention_masks = []

    for text in texts:
        encoded_dict = tokenizer.encode_plus(
                            text,
                            add_special_tokens=True,
                            max_length=max_length,
                            padding='max_length',
                            truncation=True,
                            return_attention_mask=True,
                            return_tensors='pt'
                       )
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(labels.tolist())

    return TensorDataset(input_ids, attention_masks, labels)

train_dataset = tokenize_data(X_train, y_train)
test_dataset = tokenize_data(X_test, y_test)

train_dataloader = DataLoader(train_dataset, sampler=None, batch_size=batch_size)
test_dataloader = DataLoader(test_dataset, sampler=None, batch_size=batch_size)

epochs = 1
learning_rate = 2e-5
optimizer = AdamW(model.parameters(), lr=learning_rate)

for epoch in range(epochs):
    model.train()
    total_loss = 0

    for batch in tqdm(train_dataloader, desc=f'Epoch {epoch + 1}/{epochs}'):
        batch = tuple(t.to(device) for t in batch)
        input_ids, attention_mask, labels = batch

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

    avg_loss = total_loss / len(train_dataloader)
    logging.info(f'Epoch {epoch + 1}/{epochs}, Average Loss: {avg_loss}')

model.eval()
predictions, true_labels = [], []

for batch in tqdm(test_dataloader, desc='Evaluating'):
    batch = tuple(t.to(device) for t in batch)
    input_ids, attention_mask, labels = batch

    with torch.no_grad():
        outputs = model(input_ids, attention_mask)

    logits = outputs.detach().cpu().numpy()
    label_ids = labels.cpu().numpy()
    predictions.append(logits)
    true_labels.append(label_ids)

flat_predictions = [item for sublist in predictions for item in sublist]
flat_predictions = np.argmax(flat_predictions, axis=1).flatten()
flat_true_labels = [item for sublist in true_labels for item in sublist]

class_names = [str(label) for label in balanced_df['Label'].unique().tolist()]

report = classification_report(flat_true_labels, flat_predictions, target_names=class_names)
logging.info(f"Classification Report:\n{report}")

accuracy = accuracy_score(flat_true_labels, flat_predictions)
logging.info(f"Accuracy: {accuracy:.4f}")

precision, recall, f1_score, _ = precision_recall_fscore_support(flat_true_labels, flat_predictions, average='weighted')
logging.info(f"Precision: {precision:.4f}")
logging.info(f"Recall: {recall:.4f}")
logging.info(f"F1-score: {f1_score:.4f}")

cm = confusion_matrix(flat_true_labels, flat_predictions)
fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
ax.set_title("Confusion Matrix")
ax.set_xticks(np.arange(len(class_names)))
ax.set_yticks(np.arange(len(class_names)))
ax.set_xticklabels(class_names, rotation=45)
ax.set_yticklabels(class_names)
thresh = cm.max() / 2.0
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, format(cm[i, j], 'd'), ha="center", va="center", color="white" if cm[i, j] > thresh else "black")
fig.tight_layout()
plt.savefig("confusion_matrix.png")
logging.info("Confusion matrix saved to: confusion_matrix.png")

train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

for epoch in range(epochs):
    model.train()
    train_loss = 0.0
    train_preds = []
    train_labels = []

    for batch in train_dataloader:
        batch = tuple(t.to(device) for t in batch)
        input_ids, attention_mask, labels = batch

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()

        train_preds.extend(outputs.detach().cpu().numpy())
        train_labels.extend(labels.cpu().numpy())

    train_loss /= len(train_dataloader)
    train_losses.append(train_loss)
    train_preds = np.argmax(train_preds, axis=1).flatten()
    train_accuracy = accuracy_score(train_labels, train_preds)
    train_accuracies.append(train_accuracy)

    model.eval()
    val_loss = 0.0
    val_preds = []
    val_labels = []

    for batch in test_dataloader:
        batch = tuple(t.to(device) for t in batch)
        input_ids, attention_mask, labels = batch

        with torch.no_grad():
            outputs = model(input_ids, attention_mask)

        loss = nn.CrossEntropyLoss()(outputs, labels)
        val_loss += loss.item()

        val_preds.extend(outputs.detach().cpu().numpy())
        val_labels.extend(labels.cpu().numpy())

    val_loss /= len(test_dataloader)
    val_losses.append(val_loss)
    val_preds = np.argmax(val_preds, axis=1).flatten()
    val_accuracy = accuracy_score(val_labels, val_preds)
    val_accuracies.append(val_accuracy)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(range(1, epochs+1), train_losses, label='Training Loss')
plt.plot(range(1, epochs+1), val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss Curves')

plt.subplot(1, 2, 2)
plt.plot(range(1, epochs+1), train_accuracies, label='Training Accuracy')
plt.plot(range(1, epochs+1), val_accuracies, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Accuracy Curves')

plt.tight_layout()
plt.savefig("learning_curves.png")
logging.info("Learning curves saved to: learning_curves.png")

model_path = 'scibert_model.pth'
torch.save(model.state_dict(), model_path)
logging.info(f"Model saved locally: {model_path}")

def save_model_to_gcs(model, model_path):
    client = storage.Client()
    bucket = client.bucket('scientific_paper_classifier-bucket')
    blob = bucket.blob(f'models/{model_path}')
    blob.upload_from_filename(model_path)
    logging.info(f"Model saved to GCS: gs://scientific_paper_classifier-bucket/models/{model_path}")

save_model_to_gcs(model, model_path)
loaded_model = SciBERTClassifier(num_labels)
loaded_model.load_state_dict(torch.load(model_path))
loaded_model.to(device)
logging.info("Model loaded successfully")

# Test on sample text
sample_text = "This is a sample text to test the model."

def predict_label(text, model, tokenizer):
    encoded_dict = tokenizer.encode_plus(
                        text,
                        add_special_tokens=True,
                        max_length=max_length,
                        padding='max_length',
                        truncation=True,
                        return_attention_mask=True,
                        return_tensors='pt'
                   )
    input_ids = encoded_dict['input_ids'].to(device)
    attention_mask = encoded_dict['attention_mask'].to(device)

    with torch.no_grad():
        outputs, attention_weights = model(input_ids, attention_mask)

    predicted_label = torch.argmax(outputs).item()
    return predicted_label, attention_weights

predicted_label = predict_label(sample_text, loaded_model, tokenizer)
logging.info(f"Predicted label for the sample text: {class_names[predicted_label]}")

def visualize_attention(text, attention_weights, tokenizer):
    encoded_dict = tokenizer.encode_plus(
                        text,
                        add_special_tokens=True,
                        max_length=max_length,
                        padding='max_length',
                        truncation=True,
                        return_attention_mask=True,
                        return_tensors='pt'
                   )
    input_ids = encoded_dict['input_ids'].squeeze().tolist()
    tokens = tokenizer.convert_ids_to_tokens(input_ids)

    attention_weights = attention_weights.squeeze().mean(dim=0).cpu().numpy()

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(attention_weights, cmap='viridis')
    ax.set_xticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation=90)
    ax.set_yticks(range(len(tokens)))
    ax.set_yticklabels(tokens)
    ax.set_title("Attention Visualization")
    fig.tight_layout()
    plt.show()

predicted_label, attention_weights = predict_label(sample_text, loaded_model, tokenizer)
logging.info(f"Predicted label for the sample text: {class_names[predicted_label]}")

visualize_attention(sample_text, attention_weights[-1], tokenizer)
