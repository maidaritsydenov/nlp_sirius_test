from llama_cpp import Llama

SYSTEM_PROMPT = "Ты — Сайга, русскоязычный автоматический ассистент. Ты разговариваешь с людьми и помогаешь им."
SYSTEM_TOKEN = 1788
USER_TOKEN = 1404
BOT_TOKEN = 9225
LINEBREAK_TOKEN = 13
ROLE_TOKENS = {
    "user": USER_TOKEN,
    "bot": BOT_TOKEN,
    "system": SYSTEM_TOKEN
}

class LlamaInference:
    def __init__(self, model_path: str = "models/saiga_model-q4_K.gguf"):
        self.model = Llama(model_path=model_path, n_ctx=2000, n_parts=1)

    def _get_message_tokens(self, role, content):
        message_tokens = self.model.tokenize(content.encode("utf-8"))
        message_tokens.insert(1, ROLE_TOKENS[role])
        message_tokens.insert(2, LINEBREAK_TOKEN)
        message_tokens.append(self.model.token_eos())
        return message_tokens

    def _get_system_tokens(self, llm_prompt: str):
        system_message = {
            "role": "system",
            "content": llm_prompt
        }
        return self._get_message_tokens(**system_message)

    def interact(
            self,
            user_message: str,
            llm_prompt=SYSTEM_PROMPT,
            top_k=30,
            top_p=0.9,
            temperature=0.2,
            repeat_penalty=1.1,
            max_tokens=2048
    ):
        system_tokens = self._get_system_tokens(llm_prompt)
        tokens = system_tokens
        self.model.eval(tokens)

        message_tokens = self._get_message_tokens(role="user", content=user_message)
        role_tokens = [self.model.token_bos(), BOT_TOKEN, LINEBREAK_TOKEN]
        tokens += message_tokens + role_tokens
        completion = self.model.create_completion(
            tokens,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            repeat_penalty=repeat_penalty,
            max_tokens=max_tokens
        )
        return completion["choices"][0]["text"]


if __name__ == "__main__":
    model_path = "./models/saiga_model-q4_K.gguf"
    llm_assistant = LlamaInference(model_path=model_path)
    # user_message = "У кого был арендован Россией комплекс Байконур?"
    # user_message = "Кем был ослеплен князь Василий Тёмный?"
    user_message = "Население Шымкента в прежних границах в начале 2015 года?"
    answer = llm_assistant.interact(user_message=user_message, llm_prompt=SYSTEM_PROMPT)
    print(answer)
