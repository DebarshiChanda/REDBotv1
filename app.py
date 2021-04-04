import numpy as np
import pandas as pd
import torch
from transformers import BertTokenizer
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from model import BERTClass
from flask import Flask, render_template, url_for, request

MAX_LEN = 200
device = "cuda:0"
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

model = BERTClass()
model.to(device)
model.eval()

app = Flask(__name__)


@app.route('/')
def home():
    return render_template("index.html")


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        text = request.form['text']

        inputs = tokenizer.encode_plus(
            text,
            None,
            truncation=True,
            add_special_tokens=True,
            max_length=MAX_LEN,
            padding='max_length',
            return_token_type_ids=True
        )
        ids = torch.tensor(inputs['input_ids'], dtype=torch.long).unsqueeze(dim=0).to(device)
        mask = torch.tensor(inputs['attention_mask'], dtype=torch.long).unsqueeze(dim=0).to(device)
        token_type_ids = torch.tensor(inputs["token_type_ids"], dtype=torch.long).unsqueeze(dim=0).to(device)

        model.load_state_dict(torch.load("model.bin"))

        outputs = model(ids, mask, token_type_ids)
        outputs = np.array(torch.sigmoid(outputs).detach().cpu())
        emotion2genre_matrix = np.array([[7, 10, 18, 21, 7, 59, 7, 10, 15, 23, 15, 3, 2, 11, 1, 7, 0, 3, 27, 24, 8, 10, 4, 2],
                                  [36, 26, 50, 31, 12, 53, 27, 14, 24, 28, 32, 8, 7, 20, 22, 27, 3, 7, 38, 37, 17, 32, 14, 9],
                                  [24, 21, 19, 13, 5, 53, 21, 10, 13, 12, 13, 4, 11, 5, 15, 19, 1, 3, 17, 16, 7, 19, 18, 3],
                                  [9, 14, 15, 17, 9, 52, 10, 10, 19, 23, 12, 6, 4, 1, 15, 13, 3, 3, 32, 12, 10, 16, 3, 5],
                                  [20, 24, 19, 13, 13, 48, 10, 12, 7, 15, 12, 5, 8, 4, 13, 10, 5, 7, 25, 14, 10, 16, 6, 3],
                                  [31, 14, 35, 20, 7, 35, 24, 13, 12, 13, 15, 7, 12, 10, 8, 23, 3, 7, 19, 23, 12, 30, 8, 7]])
        # emotion2genre_matrix = emotion2genre_matrix / 100.
        emotion2genre_matrix = normalize(emotion2genre_matrix, axis=1, norm='l2')

        outputs = np.matmul(outputs, emotion2genre_matrix)
        outputs = normalize(outputs, axis=1, norm='l2')

        ones = np.ones((596, 24))
        outputs = outputs * ones

        movie_df = pd.read_csv("new_movies.csv")
        genre_cols = ['Action', 'Adult', 'Adventure', 'Animation', 'Biography', 'Comedy', 'Crime', 'Documentary',
                      'Drama', 'Family', 'Fantasy', 'Film-Noir', 'History', 'Horror', 'Musical', 'Mystery', 'News',
                      'Reality-TV', 'Romance', 'Sci-Fi', 'Sport', 'Thriller', 'War', 'Western']
        vector = movie_df[genre_cols].values
        normalized_vector = normalize(vector, axis=1, norm='l2')
        similarity = cosine_similarity(outputs, normalized_vector)
        movie_df['similarity'] = similarity[0]

        movies = movie_df.sort_values(by=['similarity', 'avg_vote', 'year'], ascending=False).head(5)['original_title']
        movie_1 = movies.iloc[0]
        movie_2 = movies.iloc[1]
        movie_3 = movies.iloc[2]
        movie_4 = movies.iloc[3]
        movie_5 = movies.iloc[4]

    return render_template('results.html', movie_1=movie_1, movie_2=movie_2, movie_3=movie_3, movie_4=movie_4, movie_5=movie_5)


if __name__ == '__main__':
    app.run()
