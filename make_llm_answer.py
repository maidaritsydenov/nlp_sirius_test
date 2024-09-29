from llama_cpp import Llama

SYSTEM_PROMPT = "Ты — Сайга, русскоязычный автоматический ассистент. Ты разговариваешь с людьми и помогаешь им."


class LlamaInference:
    def __init__(self, model_path: str = "models/saiga_llama3_q4/model-q4_K.gguf"):
        self.model = Llama(model_path=model_path, n_ctx=8192, n_parts=1, verbose=False)

    def interact(
            self,
            user_message: str,
            top_k=30,
            top_p=0.9,
            temperature=0.6,
            repeat_penalty=1.1
    ):
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.append({"role": "user", "content": user_message})

        llm_answer = self.model.create_chat_completion(
            messages,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repeat_penalty=repeat_penalty,
            stream=False,
            max_tokens=2048,
        )["choices"][0]["message"]["content"]
        return llm_answer

# if __name__ == "__main__":
#     model_path = "models/saiga_llama3_q4/model-q4_K.gguf"
#     llm_assistant = LlamaInference(model_path=model_path)
#     # user_message = "У кого был арендован Россией комплекс Байконур?"
#     # user_message = "Население Шымкента в прежних границах в начале 2015 года?"
#     user_message = "Кем был ослеплен князь Василий Тёмный?"
#     answer = llm_assistant.interact(user_message)
#     print(answer)