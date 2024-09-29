import os
from typing import Dict
from tqdm.auto import tqdm
import pandas as pd
import numpy as np
import torch
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
import pickle

tqdm.pandas()


class CreateEmbeddings:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-base")
        self.model = AutoModel.from_pretrained("intfloat/multilingual-e5-base").to(self.device)

    def _average_pool(self, last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    def get_embeddings(self, data_dict: Dict[str, str]):
        embeddings_raw_name = './db_embedds/embeddings.npy'
        embeddings_question_name = './db_embedds/embeddings.txt'
        embeddings_answer_name = './db_embedds/embeddings_answer.txt'

        if not os.path.exists(embeddings_raw_name):
            print("Генерация эмбеддингов.")
            questions = ["passage: " + q for q in list(data_dict.keys())]
            answers = list(data_dict.values())
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
            questions_list = questions
            answers_list = answers
        else:
            print("Загрузка готовых эмбеддингов.")
            embeddings_raw = np.load(embeddings_raw_name)
            with open(embeddings_question_name, 'r', encoding='utf-8') as f:
                questions_list = f.readlines()
            with open(embeddings_answer_name, 'r', encoding='utf-8') as f:
                answers_list = f.readlines()

        return embeddings_raw, questions_list, answers_list

    def get_embedding(self, text: str):
        if "passage" not in text:
            text = "passage: " + text
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


if __name__ == "__main__":

    if not os.path.exists("./db_embedds/data_dict.pickle"):
        train = pd.read_parquet("./dataset/kuznetsoffandrey_sberquad_train.parquet")
        valid = pd.read_parquet("./dataset/kuznetsoffandrey_sberquad_valid.parquet")
        test = pd.read_parquet("./dataset/kuznetsoffandrey_sberquad_test.parquet")

        all_data = pd.concat([train, valid], axis=0)
        all_data = pd.concat([all_data, test], axis=0)

        data_dict = {}

        def get_data_dict(row):
            question = row["question"]
            answer = row["answers"]["text"][0]
            data_dict[str(question)] = str(answer)

        all_data.progress_apply(get_data_dict, axis=1)
        data_dict = {k: v for k, v in data_dict.items() if v not in ["", " ", None]}

        with open("./db_embedds/data_dict.pickle", "wb") as f:
            pickle.dump(data_dict, f)

    else:
        data_dict = pickle.load(open("./db_embedds/data_dict.pickle", "rb"))

    embedds_model = CreateEmbeddings()
    embeddings_raw, questions_list, answers_list = embedds_model.get_embeddings(data_dict)
    print(embeddings_raw[0][:30])
    print(questions_list[0][:30])
    print(answers_list[0][:30])