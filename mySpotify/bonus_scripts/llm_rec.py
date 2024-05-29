import os
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import CSVLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import OpenAI
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
import streamlit as st
import logging
from colorama import Fore, Style, init
import gradio as gr
import time

from loguru import logger
from typing import Optional, List
from pydantic import BaseModel

import os
import polars as pl
import pandas as pd



class ColorLogFormatter(logging.Formatter):
    def __init__(self):
        super().__init__('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    def format(self, record):
        log_colors = {
            'DEBUG': Fore.CYAN,
            'INFO': Fore.GREEN,
            'WARNING': Fore.YELLOW,
            'ERROR': Fore.RED,
            'CRITICAL': Fore.RED + Style.BRIGHT
        }
        log_color = log_colors.get(record.levelname, Fore.WHITE)
        record.msg = log_color + str(record.msg) + Style.RESET_ALL
        return super().format(record)

# Configure logging
logger = logging.getLogger()
handler = logging.StreamHandler()
handler.setFormatter(ColorLogFormatter())
logger.setLevel(logging.INFO)
logger.addHandler(handler)



api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
  raise ValueError("Please set the OPENAI_API_KEY environment variable")

global_path = './data'
triplet_path = f"{global_path}/train_triplets.txt"
unique_tracks_path = f"{global_path}/p02_unique_tracks.txt"
genre_path = f"{global_path}/p02_msd_tagtraum_cd2.cls"


class MusicData:
    def __init__(self, data):
        self.data = data
        self.song_id_to_name = pd.Series(data.song.values, index=data.song_id).to_dict()

    def get_user_song_names(self, user_id):
        user_data = self.data[self.data['user_id'] == user_id]
        user_songs = [self.song_id_to_name[song_id] for song_id in user_data['song_id'].unique()]
        return "\n".join([f"- {song}" for song in user_songs])

    def get_user_song_ids(self, user_id):
      user_data = self.data[self.data['user_id'] == user_id]
      return user_data['song_idx'].unique()


def load_data(triplet_path, unique_tracks_path):
    logging.info('Loading data...')


    triplet_columns = ['user_id', 'song_id', 'play_count']
    track_columns = ['track_id', 'song_id', 'artist', 'title']

    triplet_df = pl.read_csv(triplet_path, separator='\t', new_columns=triplet_columns, use_pyarrow=True)
    unique_tracks_df = pl.from_pandas(pd.read_csv(unique_tracks_path, names=track_columns, sep="<SEP>", engine='python'))

    logging.info('Data loaded successfully.')

    logging.info('Merging songs...')

    triplet_df = triplet_df.filter(pl.col('play_count') > 1)
    songs = pd.merge(triplet_df.to_pandas(), unique_tracks_df.to_pandas(), on='song_id', how='left')
    songs['song'] = songs['title']+' - ' + songs['artist']
    songs = songs[['user_id', 'track_id', 'song_id', 'song', 'play_count']]

    logging.info('Songs merged successfully.')

    del triplet_df, unique_tracks_df

    return songs

def get_heard_of(user_id: str, music_data: MusicData) -> str:
  return music_data.get_user_song_names(user_id)


def llm_rec():
    # loader = CSVLoader(file_path="llm_rec_test.csv")
    loader = CSVLoader(file_path=f"{global_path}/llm_RecSys_dataset_updated.csv")
    data = loader.load()

    logging.info(f"Loaded {len(data)} documents")

    #data transformers
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(data)

    logging.info(f"Split {len(texts)} documents")

    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    llm = ChatOpenAI(model="gpt-4o", openai_api_key=api_key)

    #Vector DB
    logging.info("Creating vector database")
    if not os.path.exists("./vector_db"):
        os.makedirs("./vector_db")
        docsearch = Chroma.from_documents(texts, embeddings, persist_directory="./vector_db")
    else:
        docsearch = Chroma(persist_directory="./vector_db", embedding_function=embeddings)
    logging.info("Vector database created")

    return llm, docsearch


def create_prompt_template(user_id, heard_of, age, gender):
    template_prefix = """You are a music recommender system that helps users find songs that match their preferences.
    Use the following pieces of context to answer the question at the end.
    For each question, suggest five songs, with a short description of the song's genre, mood, and the reason why the user might like it.
    For each question, take into account the context and the personal information provided by the user.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.

    {context}"""

    user_info = """This is what we know about the user, and you can use this information (age, gender and listening history) to better tune your research:
    User ID: {user_id}
    Age: {age}
    Gender: {gender}
    User: {user_id} has listened the following songs: {heard_of}
    """

    template_suffix= """Question: {question}
    Your response:"""

    user_info = user_info.format(
      user_id=user_id,
      age=18,
      gender='female',
      heard_of=heard_of
    )

    combind_prompt = template_prefix +'\n'+ user_info +'\n'+ template_suffix

    prompt = PromptTemplate(
        template=combind_prompt, input_variables=["context", "question"])

    chain_type_kwargs = {"prompt": prompt}
    qa = RetrievalQA.from_chain_type(llm=llm,
        chain_type="stuff",
        retriever=docsearch.as_retriever(),
        return_source_documents=True,
        chain_type_kwargs=chain_type_kwargs)

    return qa


if __name__ == "__main__":
    music_data = MusicData(load_data(triplet_path, unique_tracks_path))
    llm, docsearch  = llm_rec()

    with gr.Blocks() as demo:
        chatbot = gr.Chatbot()
        msg = gr.Textbox(label="Message")
        user_id_input = gr.Textbox(label="User ID", value='b7815dbb206eb2831ce0fe040d0aa537e2e800f7')
        age_input = gr.Number(label="Age", value=18)
        gender_input = gr.Dropdown(label="Gender", choices=["male", "female", "other"], value="male")
        clear = gr.ClearButton([msg, chatbot, user_id_input, age_input, gender_input])
        chat_history = []

        def respond(message, chat_history, user_id, age, gender):
            qa = create_prompt_template(user_id=user_id, age=age, gender=gender, heard_of=get_heard_of(user_id, music_data))
            bot_message = qa.invoke({'query': message, 'chat_history': chat_history})
            chat_history.append((message, bot_message['result']))
            return "", chat_history

        def vote(data: gr.LikeData):
            if data.liked:
                print("You upvoted this response: " + data.value)
            else:
                print("You downvoted this response: " + data.value)

        msg.submit(respond, [msg, chatbot, user_id_input, age_input, gender_input], [msg, chatbot])
        chatbot.like(vote, None, None)

    demo.launch()
