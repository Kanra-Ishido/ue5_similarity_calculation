import numpy as np
import torch
from fastapi import FastAPI
from transformers import BertForSequenceClassification, BertTokenizer

app = FastAPI()
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased", #英語のbert-base-uncasedモデルの指定
    num_labels = 2, # ラベル数（今回はBinayなので2、数値を増やせばマルチラベルも対応可）
    output_attentions = False, # アテンションベクトルを出力するか
    output_hidden_states = True, # 隠れ層を出力するか
)
model.eval()

def get_embedding(word, tokenizer, model):
    sent = word
    tokenized_text = tokenizer.tokenize(sent)

    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

    tokens_tensor = torch.tensor([indexed_tokens])
    with torch.no_grad(): # 勾配計算なし
    
        all_encoder_layers = model(tokens_tensor)

    embedding = all_encoder_layers[1][-2].numpy()[0]
    t = np.mean(embedding, axis=0)

    t = t.reshape(1, 768)

    return t

@app.get("/")
async def rooot():
    return("hello")


@app.post("/similarity_calculation")
async def similarity_calculation(GPT_word, file_name):
    vec_GPT_word = get_embedding(GPT_word, tokenizer, model)
    vec_file_name = get_embedding(file_name, tokenizer, model)

    dot_product = np.dot(vec_GPT_word, vec_file_name.T)
    norm_vec1 = np.linalg.norm(vec_GPT_word)
    norm_vec2 = np.linalg.norm(vec_file_name)
    similarity = dot_product / (norm_vec1 * norm_vec2)

    return (similarity[0][0]).item()

@app.post("/test")
async def test(hoge):
    return(hoge)