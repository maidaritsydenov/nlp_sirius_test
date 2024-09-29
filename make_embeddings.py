import os
import pickle
from typing import Dict
from tqdm.auto import tqdm
import numpy as np
import torch
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
from sklearn.feature_extraction.text import TfidfVectorizer


tqdm.pandas()


class CreateEmbeddings:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-base")
        self.model = AutoModel.from_pretrained("intfloat/multilingual-e5-base").to(self.device)
        self.vectorizer = TfidfVectorizer(max_features=2000, lowercase=True, analyzer="word")

    def _average_pool(self, last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    def get_embeddings(self, data_dict: Dict[str, str]):
        embeddings_raw_name = './db_embedds/embeddings.npy'
        embeddings_question_name = './db_embedds/embeddings.txt'
        embeddings_answer_name = './db_embedds/embeddings_answer.txt'
        tfidf_matrix_name = './db_embedds/tfidf_matrix.npy'
        tfidf_vectorizer_path = './db_embedds/tfidf_vectorizer.pickle'

        questions = ["passage: " + q for q in list(data_dict.keys())]
        answers = list(data_dict.values())

        if not os.path.exists(embeddings_raw_name):
            print("Генерация эмбеддингов.")
            question_embeddings = []

            with torch.no_grad():
                for question in tqdm(questions, desc="generating embedds"):
                    batch_dict = self.tokenizer(
                        question,
                        max_length=512,
                        padding=True,
                        truncation=True,
                        return_tensors="pt",
                    ).to(self.device)
                    outputs = self.model(**batch_dict)
                    embedding = self._average_pool(
                        outputs.last_hidden_state, batch_dict["attention_mask"]
                    ).cpu()
                    question_embeddings.append(embedding[0])
                question_embeddings = torch.stack(question_embeddings).cpu().detach().numpy()

            np.save(embeddings_raw_name, question_embeddings)
            with open(embeddings_question_name, 'w', encoding='utf-8') as f:
                for line in tqdm(questions, desc="saving questions"):
                    f.write(line + '\n')
            with open(embeddings_answer_name, 'w', encoding='utf-8') as f:
                for line in tqdm(answers, desc="saving answers"):
                    f.write(line + '\n')

            embeddings_raw = question_embeddings

        else:
            print("Загрузка готовых эмбеддингов.")
            embeddings_raw = np.load(embeddings_raw_name)
            with open(embeddings_question_name, 'r', encoding='utf-8') as f:
                questions = f.readlines()
            with open(embeddings_answer_name, 'r', encoding='utf-8') as f:
                answers = f.readlines()

        if not os.path.exists(tfidf_vectorizer_path):
            # Генерация и сохранение TF-IDF
            print("Генерация TF-IDF.")
            tfidf_matrix = self.vectorizer.fit_transform(questions).toarray()
            np.save(tfidf_matrix_name, tfidf_matrix)
            with open(tfidf_vectorizer_path, 'wb') as f:
                pickle.dump(self.vectorizer, f)

        else:
            print("Загрузка готовых эмбеддингов TF-IDF.")
            tfidf_matrix = np.load(tfidf_matrix_name)
            with open(tfidf_vectorizer_path, 'rb') as f:
                self.vectorizer = pickle.load(f)

        return embeddings_raw, questions, answers, tfidf_matrix


    def get_embedding(self, text: str):
        text = "passage: " + text if "passage" not in text else text
        batch_dict = self.tokenizer(
            text, max_length=512, padding=True, truncation=True, return_tensors="pt"
        )
        outputs = self.model.to("cpu")(**batch_dict)
        embedding = (
            self._average_pool(outputs.last_hidden_state, batch_dict["attention_mask"])
            .cpu()
            .detach()
            .numpy()
        )
        return embedding

    def get_tfidf(self, text: str):
        return self.vectorizer.transform([text]).toarray()
