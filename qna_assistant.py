import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from make_embeddings import CreateEmbeddings
from make_llm_answer import LlamaInference


DATA_DICT_PATH = "./db_embedds/data_dict.pickle"
EMBEDDS_W = 0.7
TFIDF_W = 0.3

class QnA_assistant:
    def __init__(
            self,
            embedds_model,
            llm_assistant,
            embeddings_raw,
            questions_list,
            answers_list,
            tfidf_matrix,
            threshold: float = 0.92):

        self.embedds_model: CreateEmbeddings = embedds_model
        self.llm_assistant: LlamaInference = llm_assistant
        self.embeddings_raw = embeddings_raw
        self.questions_list = questions_list
        self.answers_list = answers_list
        self.tfidf_matrix = tfidf_matrix
        self.threshold = threshold
        print("DEBUG INFO: threshold:", self.threshold)

    def get_answer(self, text: str) -> str:
        print(f"INFO: user question: {text}")
        sorted_index, scores = self.calc_combined_similarity(text)
        best_match_idx = sorted_index[0]
        cos_score = scores[best_match_idx]

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

    def calc_combined_similarity(self, text: str):
        embedding = self.embedds_model.get_embedding(text)
        tfidf_vector = self.embedds_model.get_tfidf(text)

        scores_embedds = []
        scores_tfidf = []

        print('DEBUG INFO: get cosine similarity bert')
        cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        for embed in self.embeddings_raw:
            score = cos(torch.Tensor(embedding), torch.Tensor(embed))
            scores_embedds.append(score[0].item())

        print('DEBUG INFO: get cosine similarity tf-idf')
        scores_tfidf = cosine_similarity(tfidf_vector, self.tfidf_matrix).flatten()

        # Комбинируем
        combined_scores = EMBEDDS_W * np.array(scores_embedds) + TFIDF_W * np.array(scores_tfidf)
        sorted_index = np.argsort(combined_scores)[::-1]

        return sorted_index, combined_scores
