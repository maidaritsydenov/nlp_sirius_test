import os
import logging
import pickle
import pandas as pd
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters
from make_embeddings import CreateEmbeddings
from make_llm_answer import LlamaInference
from qna_assistant import QnA_assistant

# Настраиваем логирование
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO
)
logger = logging.getLogger(__name__)

# Пути к данным
DATA_DICT_PATH = "./db_embedds/data_dict.pickle"

# Загрузка и подготовка данных и моделей
if not os.path.exists(DATA_DICT_PATH):
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

    with open(DATA_DICT_PATH, "wb") as f:
        pickle.dump(data_dict, f)

else:
    data_dict = pickle.load(open(DATA_DICT_PATH, "rb"))


embedds_model = CreateEmbeddings()
embeddings_raw, questions_list, answers_list, tfidf_matrix  = embedds_model.get_embeddings(data_dict)
llm_assistant = LlamaInference()
assistant = QnA_assistant(
    embedds_model,
    llm_assistant,
    embeddings_raw,
    questions_list,
    answers_list,
    tfidf_matrix,
    threshold=0.9
)

async def start(update: Update, context) -> None:
    user = update.effective_user
    await update.message.reply_text(f"Привет, {user.first_name}! Задай мне любой вопрос, и я постараюсь ответить!")

async def handle_message(update: Update, context) -> None:
    user_question = update.message.text
    logger.info(f"Получен вопрос: {user_question}")
    # Получение ответа на вопрос
    answer = assistant.get_answer(user_question)
    # Ответ пользователю
    await update.message.reply_text(answer)


def main() -> None:
    BOT_TOKEN = "6068998865:AAG3wELNimQPbkiuuIfCTzHmNXo-J7ki9Yw"
    application = Application.builder().token(BOT_TOKEN).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    application.run_polling()


if __name__ == "__main__":
    main()
