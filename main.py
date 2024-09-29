import os
import pickle
import pandas as pd
import numpy as np
import torch
from make_embeddings import CreateEmbeddings
from make_llm_answer import LlamaInference

DATA_DICT_PATH = "./db_embedds/data_dict.pickle"

class QnA_assistant:
    def __init__(
            self,
            embedds_model,
            llm_assistant,
            embeddings_raw,
            questions_list,
            answers_list,
            threshold: float = 0.92):

        self.embedds_model: CreateEmbeddings = embedds_model
        self.llm_assistant: LlamaInference = llm_assistant
        self.embeddings_raw = embeddings_raw
        self.questions_list = questions_list
        self.answers_list = answers_list
        self.threshold = threshold

    def main(self, text: str) -> str:
        print(f"INFO: user question: {text}")
        sorted_index, scores = self.calc_cos_similarity(text)
        best_match_idx = sorted_index[0]
        cos_score = scores[best_match_idx].numpy()

        # Если есть похожий вопрос
        print(f"INFO: {float(cos_score) = }")
        if float(cos_score) >= self.threshold:
            # Возвращаем ответ на этот вопрос
            _, supposed_answer = self.questions_list[best_match_idx], self.answers_list[best_match_idx]
            print(f"INFO: answer: {supposed_answer}")
            return supposed_answer
        else:
            # Иначе создаем промпт для llm
            llm_prompt = self._get_prompt(text, sorted_index)
            print(f"INFO: llm_prompt:\n{llm_prompt}")
            # Генерим ответ с помощью llm
            llm_answer = self.llm_assistant.interact(user_message=llm_prompt)
            print(f"INFO: answer: {llm_answer}")
            return llm_answer

    def _get_prompt(self, text: str, sorted_index: np.ndarray) -> str:
        """Часть хорошего промпт-инжиниринга."""
        # Sub-instruction for specific task
        llm_prompt = "Твоя задача - дать ответ на вопрос. Отвечай на русском языке.\n"
        # Retrieval-Augmented Generation, 4 nearest examples from embedding database + solutions
        llm_prompt = llm_prompt + "Следуй примерам question - answer:\n"
        for i in range(0, 3):
            question = str.replace(self.questions_list[sorted_index[i]], "\n", "")
            question = str.replace(question, "passage: ", "")
            answer = str.replace(self.answers_list[sorted_index[i]], "\n", "")

            if answer not in ["", " ", None]:
                llm_prompt = llm_prompt + f"question: {question}, answer: {answer}\n"

        # User question
        llm_prompt = llm_prompt + f"Дай ответ на вопрос:\nquestion: {text} answer: "
        return llm_prompt

    def calc_cos_similarity(self, text: str):
        embedding = self.embedds_model.get_embedding(text)
        scores = []
        cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        for embed in self.embeddings_raw:
            score = cos(torch.Tensor(embedding), torch.Tensor(embed))
            scores.append(score[0])
        sorted_index = np.argsort(scores)[::-1]
        return sorted_index, scores


# if __name__ == "__main__":
#
#     if not os.path.exists(DATA_DICT_PATH):
#         train = pd.read_parquet("./dataset/kuznetsoffandrey_sberquad_train.parquet")
#         valid = pd.read_parquet("./dataset/kuznetsoffandrey_sberquad_valid.parquet")
#         test = pd.read_parquet("./dataset/kuznetsoffandrey_sberquad_test.parquet")
#
#         all_data = pd.concat([train, valid], axis=0)
#         all_data = pd.concat([all_data, test], axis=0)
#
#         data_dict = {}
#
#         def get_data_dict(row):
#             question = row["question"]
#             answer = row["answers"]["text"][0]
#             data_dict[str(question)] = str(answer)
#
#         all_data.progress_apply(get_data_dict, axis=1)
#         data_dict = {k: v for k, v in data_dict.items() if v not in ["", " ", None]}
#
#         with open("./db_embedds/data_dict.pickle", "wb") as f:
#             pickle.dump(data_dict, f)
#
#     else:
#         data_dict = pickle.load(open(DATA_DICT_PATH, "rb"))
#
#     embedds_model = CreateEmbeddings()
#     assistant = QnA_assistant()
#     llm_assistant = LlamaInference()
#
#     embeddings_raw, questions_list, answers_list = embedds_model.get_embeddings(data_dict)
#
#     user_text = "Как улучшить свою харизму?"
#     answer = assistant.main(user_text)
#     print(answer)
