import json
import torch
from transformers import BertTokenizer
import torch.nn as nn
from transformers import BertModel
from sklearn.preprocessing import MultiLabelBinarizer


class BertForMultiLabelClassification(nn.Module):
    def __init__(self, num_labels):
        super(BertForMultiLabelClassification, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits


def load_model(path, num_labels):
    model = BertForMultiLabelClassification(num_labels)
    model.load_state_dict(torch.load(path)['model_state_dict'])
    model.eval()
    return model


def get_labels_from_dataset(file_path):
    all_labels = []
    with open(file_path, 'r') as file:
        for line in file:
            data = json.loads(line)
            all_labels.append(data['classifications'])

    mlb = MultiLabelBinarizer()
    mlb.fit(all_labels)
    return mlb.classes_


def predict_labels(title, model, tokenizer, mlb_classes, top_k=5):
    # Tokenize the input title
    encoding = tokenizer.encode_plus(
        title,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    input_ids = encoding['input_ids']
    attention_mask = encoding['attention_mask']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    model = model.to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        probs = torch.sigmoid(outputs)

    # 5 predictions
    top_k_probs, top_k_indices = torch.topk(probs, top_k, dim=1)

    predicted_labels = [mlb_classes[idx] for idx in top_k_indices[0].tolist()]

    return predicted_labels

def process_titles(input_file, output_file, model, tokenizer, mlb_classes):
    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
        for line in infile:
            title = line.strip()
            if title:
                predicted_labels = predict_labels(title, model, tokenizer, mlb_classes)
                result = {
                    "title": title,
                    "classifications": predicted_labels
                }
                outfile.write(json.dumps(result, ensure_ascii=False) + '\n')


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# getting unique labels
mlb_classes = get_labels_from_dataset('train.txt')

num_labels = len(mlb_classes)
model = load_model("bert_model.pth", num_labels)

input_file = 'test-without-classifications.txt'
output_file = 'solutions.txt'

process_titles(input_file, output_file, model, tokenizer, mlb_classes)