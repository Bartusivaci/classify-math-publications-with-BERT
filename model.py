import json
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MultiLabelBinarizer
from transformers import BertTokenizer
import torch.nn as nn
from transformers import BertModel
import torch.optim as optim
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import os

class TextClassificationDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=128, max_samples=1000000):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.texts = []
        self.labels = []

        with open(file_path, 'r') as file:
            for i, line in enumerate(file):
                if i >= max_samples:
                    break
                data = json.loads(line)
                self.texts.append(data['title'])
                self.labels.append(data.get('classifications', []))

        # multi-label conversion using MultiLabelBinarizer
        self.mlb = MultiLabelBinarizer()
        self.mlb.fit(self.labels)
        self.labels = self.mlb.transform(self.labels)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        labels = self.labels[idx]

        # tokenizing the text
        encoding = self.tokenizer.encode_plus(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': torch.tensor(labels, dtype=torch.float)
        }


# BERT classification model
class BertForMultiLabelClassification(nn.Module):
    def __init__(self, num_labels):
        super(BertForMultiLabelClassification, self).__init__()
        # loading pre-trained model
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        # adding a dropout layer
        self.dropout = nn.Dropout(0.3)
        # adding a linear layer
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        # passing inputs
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        # using dropout
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits


# training function
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, start_epoch=0, num_epochs=5, save_path="bert_model.pth"):
    for epoch in range(start_epoch, num_epochs):
        model.train()
        total_loss = 0

        # training loop
        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{num_epochs}"):
            optimizer.zero_grad()

            # changing device for GPU
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # forward
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)

            # backward
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}, Training Loss: {avg_train_loss:.4f}")

        # saving model
        save_model(model, optimizer, epoch, path=save_path)
        print(f"Model saved after epoch {epoch + 1}")

def save_model(model, optimizer, epoch, path="bert_model.pth"):
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    torch.save(state, path)
    print(f"Model saved to {path}")


def load_model(model, optimizer, path="bert_model.pth"):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    print(f"Model loaded from {path}, resuming from epoch {epoch + 1}")
    return model, optimizer, epoch


print(torch.cuda.is_available())

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

train_dataset = TextClassificationDataset('train.txt', tokenizer)
val_dataset = TextClassificationDataset('validation.txt', tokenizer)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

for batch in train_loader:
    print(batch)
    break

num_labels = len(train_dataset.mlb.classes_)
model = BertForMultiLabelClassification(num_labels)

# GPU must be used but if not available then CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

print(model)


criterion = nn.BCEWithLogitsLoss()
optimizer = optim.AdamW(model.parameters(), lr=2e-5, weight_decay=1e-2)

total_steps = len(train_loader) * 5  # epochs
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps
)

checkpoint_path = "bert_model.pth"
if os.path.exists(checkpoint_path):
    model, optimizer, start_epoch = load_model(model, optimizer, path=checkpoint_path)
else:
    print(f"No saved model was found at {checkpoint_path}. Starting training from scratch.")
    start_epoch = 0

train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, start_epoch, num_epochs=5)

