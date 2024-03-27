from flask import Flask, render_template, request, send_file, send_from_directory, jsonify
import torch
import torch.nn as nn
import pickle
from script import bert
import spacy
import pickle
import re
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

app = Flask(__name__, template_folder='templates')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = bert.BERT().to(device)
model.load_state_dict(torch.load('../model/SBERT.pth'))

pretrained_model = SentenceTransformer("all-MiniLM-L6-v2")

tokenizer = spacy.load("en_core_web_sm")
word2id = pickle.load(open('../model/elements/word2id.pkl', 'rb'))

batch_size, max_len = 8, 512

# define mean pooling function
def mean_pool(token_embeds, attention_mask):
    # reshape attention_mask to cover 768-dimension embeddings
    in_mask = attention_mask.unsqueeze(-1).expand(
        token_embeds.size()
    ).float()
    # perform mean-pooling but exclude padding tokens (specified by in_mask)
    pool = torch.sum(token_embeds * in_mask, 1) / torch.clamp(
        in_mask.sum(1), min=1e-9
    )
    pool = pool.to(device)

    return pool

def my_tokenizer(sent):
    max_seq_length = 512
    tokens = tokenizer(re.sub("[.,!?\\-']=", ' ', sent.lower()))
    input_ids = [word2id['[CLS]']] + [word2id[str(token)] for token in tokens if str(token) in word2id] + [word2id['[SEP]']]
    pad_len = max_seq_length - len(input_ids)
    attn_mask = ([1] * len(input_ids)) + ([0] * pad_len)
    input_ids += [word2id['[PAD]']] * pad_len

    input_ids_tensor = torch.tensor(input_ids).unsqueeze(0).to(device)
    attn_mask_tensor = torch.tensor(attn_mask).unsqueeze(0).to(device)

    return input_ids_tensor, attn_mask_tensor

def calculate_similarity(model, sentence_a, sentence_b, device):
    # Tokenize and convert sentences to input IDs and attention masks
    inputs_ids_a, attention_a = my_tokenizer(sentence_a)
    inputs_ids_b, attention_b = my_tokenizer(sentence_b)

    # Extract token embeddings from BERT
    segment_ids = torch.zeros(batch_size, max_len, dtype=torch.int32).to(device)

    u = model.last_hidden_state(inputs_ids_a, segment_ids) # all token embeddings A = batch_size, seq_len, hidden_dim
    v = model.last_hidden_state(inputs_ids_b, segment_ids) # all token embeddings B = batch_size, seq_len, hidden_dim

    # Get the mean-pooled vectors
    u = mean_pool(u, attention_a).detach().cpu().numpy().reshape(-1)  # batch_size, hidden_dim
    v = mean_pool(v, attention_b).detach().cpu().numpy().reshape(-1)  # batch_size, hidden_dim

    # Calculate cosine similarity
    similarity_score = cosine_similarity(u.reshape(1, -1), v.reshape(1, -1))[0, 0]

    return similarity_score

def calculate_similarity2(model, sentence_a, sentence_b):
    encoded_a = model.encode(sentence_a)
    encoded_b = model.encode(sentence_b)
    return cosine_similarity(encoded_a.reshape(1, -1), encoded_b.reshape(1, -1))[0, 0]

def similarity_to_label(similarity):
    similarity = float(similarity)
    if similarity < 1/3:
        return 'Contradiction'
    elif similarity < 2/3:
        return 'Neutral'
    else:
        return 'Entailment'
    
@ app.route('/', methods=['POST', 'GET'])
def main():
    if request.method == 'GET':
        return render_template('index.html')

@ app.route('/ifsim', methods=['POST', 'GET'])
def predict():
    if request.method == 'GET':
        return render_template('index.html')
    if request.method == 'POST':
        inputData = request.json
        p = inputData['input_a'] 
        h = inputData['input_b'] 

        print('>>>>> waking up the interpreter <<<<<')

        model.eval()

        my_similarity = calculate_similarity(model, p, h, device)
        pt_similarity = calculate_similarity2(pretrained_model, p, h)

        my_sim = similarity_to_label(my_similarity)
        pt_sim = similarity_to_label(pt_similarity)

        return jsonify({'my_sim': my_sim,
                        'pt_sim': pt_sim})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)

