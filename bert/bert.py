 # dataset.csv dataset olarak seçildiğinde classification için F1: 0.6838  Accuracy: 0.6809


import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset
from sklearn.metrics import f1_score, accuracy_score

def custom_data_collator(features):
    batch = {
        'input_ids': torch.stack([f[0] for f in features]),
        'attention_mask': torch.stack([f[1] for f in features]),
        'labels': torch.tensor([f[2] for f in features])
    }
    return batch

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    return {
        'f1': f1_score(labels, preds, average='weighted'),  # ağırlıklı F1 skoru
        'accuracy': accuracy_score(labels, preds)  # doğruluk
    }

def load_and_clean_data(file_path):
    try:
        df = pd.read_csv(file_path, encoding='utf-8', on_bad_lines='skip')
    except UnicodeDecodeError:
        df = pd.read_csv(file_path, encoding='ISO-8859-1', on_bad_lines='skip')
    except pd.errors.ParserError as e:
        print(f"CSV parse hatası: {e}")
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for i, line in enumerate(lines[:10]):
                print(f"Line {i}: {line}")
        raise

    print("Sütun Adları:", df.columns)
    if 'Görüs' not in df.columns or 'Durum' not in df.columns:
        raise ValueError("Beklenen sütunlar bulunamadı")

    df = df.rename(columns=lambda x: x.strip())

    df = df.dropna(subset=['Görüs', 'Durum'])
    df = df[df['Görüs'].apply(lambda x: isinstance(x, str))]
    df = df[df['Durum'].isin(['Olumsuz', 'Tarafsız', 'Olumlu'])]

    return df

df = load_and_clean_data('dataset.csv')

texts = df['Görüs'].tolist()
labels = df['Durum'].map({'Olumsuz': 0, 'Tarafsız': 1, 'Olumlu': 2}).tolist()


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)

encodings = tokenizer(texts, truncation=True, padding=True, max_length=128, return_tensors='pt')
input_ids = encodings['input_ids']
attention_mask = encodings['attention_mask']
labels = torch.tensor(labels)

train_inputs, test_inputs, train_labels, test_labels = train_test_split(input_ids, labels, test_size=0.2, random_state=42)
train_masks, test_masks, _, _ = train_test_split(attention_mask, attention_mask, test_size=0.2, random_state=42)

# TensorDataset oluşturma
train_dataset = TensorDataset(train_inputs, train_masks, train_labels)
test_dataset = TensorDataset(test_inputs, test_masks, test_labels)

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=10,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy='epoch'
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    data_collator=custom_data_collator,
    compute_metrics=compute_metrics
)

trainer.train()

eval_results = trainer.evaluate()
print(f"F1 Skoru: {eval_results['eval_f1']:.4f}")
print(f"Doğruluk: {eval_results['eval_accuracy']:.4f}")