import os
import argparse
import pandas as pd
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.chains import RetrievalQA
# from langchain.retrievers import RetrievalQA
# from langchain.prompts import PromptTemplate
# from langchain_openai import ChatOpenAI
# from langchain import OpenAI
# from langchain.loaders import CSVLoader
# from langchain.splitters import CharacterTextSplitter
# from langchain.vectorstores import Chroma

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




import os
# import pandas as pd
# from langchain import *
# from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
# from langchain_openai import ChatOpenAI
# from langchain.prompts import PromptTemplate

class MusicRecommender:
    def __init__(self, global_path, api_key):
        self.global_path = global_path
        self.api_key = api_key
        self.data_loader = CSVLoader(file_path=f"{global_path}/llm_RecSys_dataset_updated.csv")
        self.data = self.data_loader.load()
        self.text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        self.texts = self.text_splitter.split_documents(self.data)
        self.embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        self.docsearch = Chroma.from_documents(self.texts, self.embedding_function)
        self.llm = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0)

    def create_prompt_template(self, context, question, age, gender):
      query = "I'm looking for a song similar by rapper like Eminem, 50 Cent. What could you suggest to me?"
      template_prefix = """You are a music recommender system that helps users find songs that match their preferences.
      Use the following pieces of context to answer the question at the end.
      For each question, suggest three songs, with a short description of the song's genre, mood, and the reason why the user might like it.
      For each question, take into account the context and the personal information provided by the user.
      If you don't know the answer, just say that you don't know, don't try to make up an answer.

      {context}"""

      user_info = """This is what we know about the user, and you can use this information to better tune your research:
      User ID: {user_id}
      Age: {age}
      Gender: {gender}
      Heard of: {heard_of}
      """

      template_suffix= """Question: {question}
      Your response:"""

      user_info = user_info.format(
        user_id='b7815dbb206eb2831ce0fe040d0aa537e2e800f7'
        age=18,
        gender='female',
        heard_of=get_heard_of('b7815dbb206eb2831ce0fe040d0aa537e2e800f7', music_data)))

      combind_prompt = template_prefix +'\n'+ user_info +'\n'+ template_suffix

      prompt = PromptTemplate(
          template=combind_prompt, input_variables=["context", "question"])

      chain_type_kwargs = {"prompt": prompt}
      qa = RetrievalQA.from_chain_type(llm=llm,
          chain_type="stuff",
          retriever=docsearch.as_retriever(),
          return_source_documents=True,
          chain_type_kwargs=chain_type_kwargs)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_key", type=str, required=True)
    parser.add_argument("--global_path", type=str, required=False)
    args = parser.parse_args()

    os.environ['OPENAI_API_KEY'] = args.api_key

    # Update the dataset
    o = pd.read_csv(f"{args.global_path}/llm_RecSys_dataset.csv")
    o['combined_info'] = o.apply(lambda row: f"Song ID: {row['song_id']}\n Artist : {row['artist']}\n Title : {row['title']}\n Lyrics: {row['lyrics']}.\n Genres: {row['majority_genre']}", axis=1)
    o[['combined_info']].to_csv(f"{args.global_path}/llm_RecSys_dataset_updated.csv", index=False)

    recommender = MusicRecommender(args.global_path, args.api_key)


    # Test the recommender
    query = "I'm looking for rap songs, artists like eminem and 50cent. What could you suggest to me?"
    context = ""
    age = 18
    gender = "male"
    # result = recommender.retrieve_songs(query, context, age, gender)
    # print(result['result'])
    # print(result['source_documents'])


    st.title("🦜🔗 RecommenderLLM")

    with st.sidebar:
        openai_api_key = st.text_input("OpenAI API Key", type="password")
        "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"


    def generate_response(input_text):
        result = recommender.retrieve_songs(query, context, age, gender)
        st.info(result['result'])


    with st.form("my_form"):
        text = st.text_area("Enter text:", "What are 3 key advice for learning how to code?")
        submitted = st.form_submit_button("Submit")
        if not openai_api_key:
            st.info("Please add your OpenAI API key to continue.")
        elif submitted:
            generate_response(text)
