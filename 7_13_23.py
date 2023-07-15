# 
# Note: This code to be provided and shared inthe video show notes on Github
#
# This code provides a Telegram bot that uses the OpenAI API and the Langchain 
# library to answer questions based on contents from a loaded text document. 
# The bot has the options to start a conversation, load a document, query the 
# document, and exit the conversation. It uses a FAISS database to store and 
# search the embeddings of the document.


# import essentail libraries 
import os
import logging
import traceback
import pickle
import faiss
#

#  import Telegram classes and functions to interact with the Telegram Bot API.
from telegram import Update, ReplyKeyboardMarkup, ReplyKeyboardRemove
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext, ConversationHandler
#

# import Langchain modules text document loading, splitting, creating embeddings, 
# and running question-answering chains
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
from transformers import GPT2Tokenizer

# import the TELEGRAM_BOT_TOKEN and OPENAI_API_KEY from a config.py file
from config import TELEGRAM_BOT_TOKEN, OPENAI_API_KEY

#  set up logging configuration
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.ERROR  # Only show errors, not warnings
)
#

#  Set OpenAI API key is set as an environment variable 
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
#

#  Initialize our database variable for our document embeddings.
DATABASE = None
#

#  set up some constants for our states and variables for user choices with chatbot.
CHOOSING, TYPING_REPLY, TYPING_CHOICE = range(3) # Constants for our states
reply_keyboard = [
    ["Start Conversation"],
    ["Load Document"],
    ["Query"],
    ["Exit Conversation"],
]
markup = ReplyKeyboardMarkup(reply_keyboard, one_time_keyboard=True) #Keyboard layout
#

#  The start function starts conversation, sends welcome message, and returns to Choosing State
def start(update: Update, context: CallbackContext) -> int:
    update.message.reply_text(
        "Hi! I'm an AI Assitant. What would you like to discuss?",
        reply_markup=markup,
    )
    return CHOOSING
#

#  The load function loads a document, split it into manageable chunks, 
#  create a FAISS database and stores the document embeddings,
#  it then tells the user when ready for querying and returns to Choosing State
DATABASE_FILE = "OAIv2database.pkl"
FAISS_INDEX_FILE = "OAIv2faiss_index.pkl"
# Initialize the global DOCUMENT_TITLES dictionary
DOCUMENT_TITLES = {}

def load(update: Update, context: CallbackContext) -> int:
    directory = './files'
    txt_files = [f for f in os.listdir(directory) if f.endswith('.txt')]

    global DATABASE
    DATABASE = []

    for txt_file in txt_files:
        print(f"Loading Document: {txt_file}...")
        loader = TextLoader(os.path.join(directory, txt_file))
        documents = loader.load()

        print("Splitting Document...")
        text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=10)
        docs = text_splitter.split_documents(documents)

        print("Creating embeddings and FAISS index from documents...")
        db = FAISS.from_documents(docs, OpenAIEmbeddings()) # type: ignore
        DATABASE.append(db)

        # Store the title for each document in the DATABASE
        for doc in docs:
            DOCUMENT_TITLES[txt_file] = txt_file  # Use txt_file as the key

        print("Document processing complete!")

    with open("OAIv2database.pkl", "wb") as f:
        pickle.dump(DATABASE, f)
    faiss.write_index(DATABASE[-1].index, "OAIfaissv2_index.pkl")

    update.message.reply_text("Documents loaded and ready for Q/A!")

    # After populating the DOCUMENT_TITLES dictionary, add the following print statement
    print("Contents of DOCUMENT_TITLES:")
    for title, filename in DOCUMENT_TITLES.items():
        print(f"{title}: {filename}")

    return CHOOSING

#  The query function checks prerequisistes and then sets up the query loop. It passes
#  input the "process_query" function (defined later)
def query(update: Update, context: CallbackContext) -> int:
    if DATABASE is None:
        context.bot.send_message(
            chat_id=update.effective_chat.id,
            text="Please load a document for Querying first!",
        )
        return CHOOSING

    if update.message.text == "Query":
        context.bot.send_message(
            chat_id=update.effective_chat.id,
            text="Please enter your query:",
        )
        return TYPING_REPLY

    if update.message.text == "Exit Conversation":
            context.bot.send_message(
            chat_id=update.effective_chat.id,
            text="Bye for now...",
        )
            return exit_conversation(update, context)

    query_text = update.message.text.strip() #remove and extra spaces etc

    print("Received user input:", query_text)

    if not query_text:
        query_text = f"For example: You can ask for a summary of my knowledgebase to start"

    results = process_query(query_text)
    print("Query results:", results)  # Output the results to the terminal

    context.bot.send_message(chat_id=update.effective_chat.id, text=results)

    return TYPING_REPLY  # Stay in the query loop until Exit is selected
#

#  The exit_conversation function ends the conversation and removes the reply keyboard.
def exit_conversation(update: Update, context: CallbackContext) -> int:
    try:
        update.message.reply_text(
            "Exiting the conversation.",
            reply_markup=ReplyKeyboardRemove(),
        )
    except Exception:
        traceback.print_exc()

    return ConversationHandler.END
#

#  The process_query function is defined to handle the actual querying of the 
#  FAISS database. It uses the similarity_search method from the langchain library 
#  to find the most similar documents to the user's query in the database. 
#  It then loads a question-answering chain and prompts the question with a 
#  specific directive to limit the answer to the content in the documents. 

def process_query(query_text):
    docs = []
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2') 
    for db in DATABASE:
        docs.extend(db.similarity_search(query_text, k=1))

    chain = load_qa_chain(llm=OpenAI(client="Turbo 3.5"), chain_type="stuff")

    prompt = "Limit answers to the document content, do not make up answers, Include the Policy Title and https://www.bu.edu/policies/ link, do not make up answers."
    prompted_query_text = prompt + query_text

    total_docs_tokens = 0
    for doc in docs:
        doc_str = str(doc)  # Convert doc to string
        tokens = tokenizer.tokenize(doc_str)
        total_docs_tokens += len(tokens)

    # Tokenize the prompted_query_text
    tokens = tokenizer.tokenize(prompted_query_text)
    total_prompt_tokens = len(tokens)

    # Calculate the grand total
    grand_total = total_docs_tokens + total_prompt_tokens

    print("Total tokens in docs:", total_docs_tokens)
    print("prompt + query_text ---> Token count:", total_prompt_tokens)
    print("Grand total tokens:", grand_total)

    # Truncate the tokens to fit within the 4097 limit
    if len(tokens) > 4096:  # GPT-3 has a maximum limit of 4096 tokens
        tokens = tokens[:4096]  # Keep the first 4096 tokens

    # Convert the tokens back into a string
    truncated_prompt = tokenizer.convert_tokens_to_string(tokens)

    results = chain(
        {'input_documents': docs, "question": truncated_prompt},
        return_only_outputs=True
    )

    if "I don't know" in results['output_text']:
        results['output_text'] = "I do not know... I do not have that information in my repository"
    
    answer = results['output_text']
    return answer

#  The main function sets up the bot and starts it.  It creates an Updater object to , 
#  to receive updates from the Telegram server and deliver them to a Dispatcher. 
#  A ConversationHandler is set up to process the conversation states (CHOOSING and TYPING_REPLY). 
#  For each state, different actions are taken based on the user's message. The 
#  ConversationHandler is added to the Dispatcher, and the Updater starts polling for updates. 
#  A message is printed to the console when the bot is ready.
def main() -> None:
    updater = Updater(token=TELEGRAM_BOT_TOKEN, use_context=True)
    dp = updater.dispatcher

    conv_handler = ConversationHandler(
        entry_points=[CommandHandler("start", start)],
        states={
            CHOOSING: [
                MessageHandler(Filters.regex('^Start Conversation$'), start),
                MessageHandler(Filters.regex('^Load Document$'), load),
                MessageHandler(Filters.regex('^Query$'), query),
                MessageHandler(Filters.regex('^Exit Conversation$'), exit_conversation),
            ],
            TYPING_REPLY: [
                MessageHandler(Filters.text, query),
                MessageHandler(Filters.regex('^Exit Conversation$'), exit_conversation),
            ],
            TYPING_CHOICE: [],
        },
        fallbacks=[],
    )

    dp.add_handler(conv_handler)

    updater.start_polling()

    print("\n\033[92mDoc Bot startup has completed enter /start to begin...\033[0m")
#

#  This is the entry point that checks if the script is run directly an not imported as a module,
#  and then starts the bot.
if __name__ == "__main__":
    main()
#