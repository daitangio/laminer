"""
    Util module collects the low-level functions with little digity to have their ouwn module
"""

from datetime import datetime
import glob
import pandas as pd

import click
from langchain.globals import set_debug
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Eval also HuggingFaceEmbeddings
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.vectorstores.utils import filter_complex_metadata


def build_file_prefix():
    """
    Build a simple date-based-prefix based also on current hour
    """
    today = datetime.now()
    formatted_date = today.strftime("%Y-%m-%d-%H")
    return formatted_date


class DataModel:
    """
    Used to store global settings
    """

    def __init__(self, retriever, debug_mode, question_dir, output_dir, openai_key):
        self.retriever = retriever
        self.debug_mode = debug_mode
        self.question_dir = question_dir
        self.output_dir = output_dir
        self.openai_key = openai_key


def create_retriever(question_dir):
    """
    Create a simple RAG retriever based on a set of text facts
    """
    dir_loader = DirectoryLoader(
        f"{question_dir}/facts", glob="**/*.txt", loader_cls=TextLoader
    )
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)
    docs = dir_loader.load()
    print(f"Docs#:{len(docs)}")
    chunks = text_splitter.split_documents(docs)
    chunks = filter_complex_metadata(chunks)
    vector_store = Chroma.from_documents(
        documents=chunks, embedding=FastEmbedEmbeddings()
    )
    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "k": 3,  # how much rows to extract
            "score_threshold": 0.4,
        },
    )
    return retriever


def load_file(file_name:str):
    with open(file_name, 'r') as file:
        # Read the entire content of the file into a string
        file_content = file.read()    
    return file_content;

def load_rag_quesitons(question_dir, debug):
    # Scan questions...    
    pair ={ "question":[], "answer":[] }
    for filename in glob.glob(f"{question_dir}/questions/*.txt"):
        if debug:
            print(f"Loading {filename}")
        question=load_file(filename)
        answer=load_file(filename.replace("/questions/","/answers/"))
        pair["question"].append( question)
        pair["answer"].append(answer)                
    df = pd.DataFrame(pair)
    return df


@click.group()
@click.option(
    "--debug/--no-debug", default=False, help="Enable lanchian debugger (to spot bugs)"
)
@click.option(
    "--question-dir",
    default="./data",
    envvar="LAMINER_QUESTION_DIR",
    help="Data dir containing question sets and rag facts",
)
@click.option(
    "--output-dir",
    required=True,
    default="output",
    envvar="LAMINER_OUTPUT_DIR",
    help="Destination directory for rag tests",
)
@click.option("--openai-key", envvar="OPENAI_API_KEY", help="OPENAI's API KEY")
@click.pass_context
def cli(ctx, debug, question_dir, output_dir, openai_key):
    """
    Examiner command to run rag or collect reports.

    Run

        run.sh <command> --help

    for details on every command.

    """
    if debug:
        set_debug(True)
    r = create_retriever(question_dir)
    ctx.obj = DataModel(r, debug, question_dir, output_dir, openai_key)
