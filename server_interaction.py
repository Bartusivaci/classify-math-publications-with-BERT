
"""
To use this implementation, you simply have to implement `get_classifications` such that it returns classifications.
You can then let your agent compete on the server by calling

    python3 server_interaction.py path/to/your/config.json
"""
import json
import logging

import requests
import time

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
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data = json.loads(line)
            all_labels.append(data['classifications'])

    mlb = MultiLabelBinarizer()
    mlb.fit(all_labels)
    return mlb.classes_


def predict_labels(title, model, tokenizer, mlb_classes, top_k=5):
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

    top_k_probs, top_k_indices = torch.topk(probs, top_k, dim=1)

    predicted_labels = [mlb_classes[idx] for idx in top_k_indices[0].tolist()]

    return predicted_labels

def get_classifications(titles):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    mlb_classes = get_labels_from_dataset('train.txt')

    num_labels = len(mlb_classes)
    model = load_model("bert_model.pth", num_labels)

    results = []
    for title in titles:
        predicted_labels = predict_labels(title, model, tokenizer, mlb_classes)
        results.append(predicted_labels)

    return results


def run(config_file, action_function, parallel_runs=True):
    logger = logging.getLogger(__name__)

    with open(config_file, 'r') as fp:
        config = json.load(fp)

    actions = []
    for request_number in range(51):    # 50 runs are enough for full evaluation. Running much more puts unnecessary strain on the server's database.
        logger.info(f'Iteration {request_number} (sending {len(actions)} actions)')
        # send request
        response = requests.put(f'{config["url"]}/act/{config["env"]}', json={
            'agent': config['agent'],
            'pwd': config['pwd'],
            'actions': actions,
            'single_request': not parallel_runs,
        })
        if response.status_code == 200:
            response_json = response.json()
            for error in response_json['errors']:
                logger.error(f'Error message from server: {error}')
            for message in response_json['messages']:
                logger.info(f'Message from server: {message}')

            action_requests = response_json['action-requests']
            if not action_requests:
                logger.info('The server has no new action requests - waiting for 1 second.')
                time.sleep(1)  # wait a moment to avoid overloading the server and then try again
            # get actions for next request
            actions = []
            for action_request in action_requests:
                actions.append({'run': action_request['run'], 'action': action_function(action_request['percept'])})
        elif response.status_code == 503:
            logger.warning('Server is busy - retrying in 3 seconds')
            time.sleep(3)  # server is busy - wait a moment and then try again
        else:
            # other errors (e.g. authentication problems) do not benefit from a retry
            logger.error(f'Status code {response.status_code}. Stopping.')
            break

    print('Done - 50 runs are enough for full evaluation')


if __name__ == '__main__':
    import sys
    run("config.json", get_classifications)
