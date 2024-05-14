import os
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from google.cloud import bigquery
import numpy as np
import logging
from torch.utils.tensorboard import SummaryWriter
from transformers.trainer_utils import EvalPrediction
from transformers import EarlyStoppingCallback

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")

client = bigquery.Client()

# Data Retrieval
query = """
SELECT Processed_Text,  Label
FROM data_train.processed_text_10000
"""
df = client.query(query).to_dataframe()
logging.info(f"Retrieved {len(df)} rows from BigQuery")

# Data Preprocessing and Balancing
text_column = 'Processed_Text'
label_column = 'Label'

combined_df = df
combined_df.reset_index(drop=True, inplace=True)
combined_df['Label'] = combined_df['Label'].astype('category')
logging.info(f"Preprocessed data: {combined_df.shape}")

label_counts = combined_df['Label'].value_counts()
min_category_size = label_counts.min()
balanced_df = pd.concat([
    combined_df[combined_df['Label'] == label].sample(n=min_category_size, random_state=42)
    for label in combined_df['Label'].cat.categories
])
balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
logging.info(f"Balanced dataset: {balanced_df.shape}")

X = balanced_df['Processed_Text']
y = balanced_df['Label']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
logging.info(f"Train size: {len(X_train)}, Validation size: {len(X_val)}")

# Tokenization and Dataset Creation
tokenizer = BertTokenizer.from_pretrained('allenai/scibert_scivocab_uncased', do_lower_case=True)

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
val_dataset = tokenize_data(X_val, y_val)
logging.info(f"Tokenized data: Train dataset size: {len(train_dataset)}, Validation dataset size: {len(val_dataset)}")

train_dataloader = DataLoader(
            train_dataset,
            sampler=None,
            batch_size=batch_size
        )

val_dataloader = DataLoader(
            val_dataset,
            sampler=None,
            batch_size=batch_size
        )

# Model Initialization
model = BertForSequenceClassification.from_pretrained('allenai/scibert_scivocab_uncased', num_labels=len(balanced_df['Label'].unique()))
model.to(device)
logging.info(f"Initialized SciBERT model with {model.num_parameters()} parameters")

# Fine-Tuning Parameters
epochs = 20
learning_rate = 2e-5

# Optimizer and Scheduler
optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)  # L2 regularization
total_steps = len(train_dataloader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
logging.info(f"Training parameters: Epochs: {epochs}, Learning rate: {learning_rate}, Batch size: {batch_size}")

# Early Stopping Callback
early_stopping = EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=0.0001)

# Fine-Tuning Loop with Gradient Clipping and Dropout
for epoch in range(epochs):
    model.train()
    total_loss = 0

    for batch in tqdm(train_dataloader, desc=f'Epoch {epoch + 1}/{epochs}'):
        batch = tuple(t.to(device) for t in batch)
        inputs = {'input_ids': batch[0],
                  'attention_mask': batch[1],
                  'labels': batch[2]}

        optimizer.zero_grad()
        outputs = model(**inputs)
        loss = outputs.loss
        total_loss += loss.item()
        loss.backward()

        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()

    avg_loss = total_loss / len(train_dataloader)
    logging.info(f'Epoch {epoch + 1}/{epochs}, Average Loss: {avg_loss}')

    # Validation
    model.eval()
    val_loss = 0
    val_predictions, val_true_labels = [], []

    for batch in tqdm(val_dataloader, desc='Validating'):
        batch = tuple(t.to(device) for t in batch)
        inputs = {'input_ids': batch[0],
                  'attention_mask': batch[1],
                  'labels': batch[2]}

        with torch.no_grad():
            outputs = model(**inputs)

        val_loss += outputs.loss.item()
        logits = outputs.logits
        logits = logits.detach().cpu().numpy()
        label_ids = inputs['labels'].cpu().numpy()
        val_predictions.append(logits)
        val_true_labels.append(label_ids)

    val_avg_loss = val_loss / len(val_dataloader)
    logging.info(f'Epoch {epoch + 1}/{epochs}, Validation Loss: {val_avg_loss}')

    # Early Stopping Check
    early_stopping(val_avg_loss, model)
    if early_stopping.early_stopping_reached:
        logging.info("Early stopping triggered. Stopping training.")
        break

# Evaluation
model.eval()
predictions, true_labels = [], []

for batch in tqdm(test_dataloader, desc='Evaluating'):
    batch = tuple(t.to(device) for t in batch)
    inputs = {'input_ids': batch[0],
              'attention_mask': batch[1],
              'labels': batch[2]}

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    logits = logits.detach().cpu().numpy()
    label_ids = inputs['labels'].cpu().numpy()
    predictions.append(logits)
    true_labels.append(label_ids)

# Classification Report
flat_predictions = [item for sublist in predictions for item in sublist]
flat_predictions = np.argmax(flat_predictions, axis=1).flatten()
flat_true_labels = [item for sublist in true_labels for item in sublist]

class_names = [str(label) for label in balanced_df['Label'].unique().tolist()]

report = classification_report(flat_true_labels, flat_predictions, target_names=class_names)
logging.info(f"Classification Report:\n{report}")

# Save and Load Model
model_path = 'scibert_fine_tuned_model.pth'
torch.save(model.state_dict(), model_path)
logging.info(f"Fine-tuned model saved to: {model_path}")

# Load the Fine-Tuned Model
loaded_model = BertForSequenceClassification.from_pretrained('allenai/scibert_scivocab_uncased', num_labels=len(balanced_df['Label'].unique()))
loaded_model.load_state_dict(torch.load(model_path))
loaded_model.to(device)
logging.info("Fine-tuned model loaded successfully")
