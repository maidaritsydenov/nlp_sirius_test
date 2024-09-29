import pickle
import numpy as np
import torch
from make_embeddings import CreateEmbeddings
from make_llm_answer import LlamaInference

DATA_DICT_PATH = "./db_embedds/data_dict.pickle"

def calc_cos_similarity(text: str):
    embedding = embedds_model.get_embedding(text)
    scores = []
    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    for embed in embeddings_raw:
        score = cos(torch.Tensor(embedding), torch.Tensor(embed))
        scores.append(score[0])
    sorted_index = np.argsort(scores)[::-1]
    return sorted_index, scores


class QnA_assistant:
    def __init__(
            self,
            threshold: float = 0.9):
        self.threshold = threshold

    def main(self, text: str) -> str:
        print(f"INFO: user question: {text}")
        sorted_index, scores = calc_cos_similarity(user_text)
        best_match_idx = sorted_index[0]
        cos_score = scores[best_match_idx].numpy()

        # Если есть похожий вопрос
        if cos_score >= self.threshold:
            # Возвращаем ответ на этот вопрос
            _, supposed_answer = questions_list[best_match_idx], answers_list[best_match_idx]
            print(f"INFO: {cos_score = }")
            print(f"INFO: answer: {supposed_answer}")
            return supposed_answer
        else:
            # Иначе создаем промпт для llm
            llm_prompt = self._get_prompt(text, sorted_index)
            # Генерим ответ с помощью llm
            full_llm_answer = llm_assistant.interact(user_message=llm_prompt)
            llm_answer = full_llm_answer[len(llm_prompt):]
            print(f"INFO: answer: {llm_answer}")
            return llm_answer

    def _get_prompt(self, text: str, sorted_index: np.ndarray) -> str:
        """Часть хорошего промпт-инжиниринга."""
        # Vicuna 1.1 as stated on a model card
        llm_prompt = f"A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\nUSER: "
        # Sub-instruction for specific task
        llm_prompt = llm_prompt + "Answer the question or give the solution to the problem. Answer using your own understanding of the context. Be concise and precise. Answer in Russian.\n"
        # User question
        llm_prompt = llm_prompt + f"Запрос пользователя: {text}\n"
        # Retrieval-Augmented Generation, 4 nearest examples from embedding database + solutions
        llm_prompt = llm_prompt + "Context: "
        for i in range(0, 3):
            question = str.replace(questions_list[sorted_index[i]], "\n", "")
            question = str.replace(question, "passage: ", "")
            answer = str.replace(answers_list[sorted_index[i]], "\n", "")
            story_text = f"Вопрос: {question}, Ответ: {answer}\n"
            llm_prompt = llm_prompt + story_text

        # LLM answer
        llm_prompt = str.replace(llm_prompt, "..", ".")
        llm_prompt = llm_prompt + "\nASSISTANT: "

        return llm_prompt


if __name__ == "__main__":
    embedds_model = CreateEmbeddings()
    assistant = QnA_assistant()
    llm_assistant = LlamaInference()

    data_dict = pickle.load(open(DATA_DICT_PATH, "rb"))
    embeddings_raw, questions_list, answers_list = embedds_model.get_embeddings(data_dict)

    user_text = "Из чего появилось графито-углистое вещество?"
    answer = assistant.main(user_text)
    print(answer)
