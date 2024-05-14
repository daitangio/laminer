#!/usr/bin/env python
"""
Run module is meant to be run as a command line
"""


from datetime import datetime
import re
import click

import pandas as pd
from tqdm import tqdm

from langchain_community.chat_models import ChatOllama
# Eval also HuggingFaceEmbeddings
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.vectorstores.utils import filter_complex_metadata

from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.globals import set_debug


def build_file_prefix():
    """
    Build a simple date-based-prefix based also on current hour
    """
    today = datetime.today().date()
    formatted_date = today.strftime('%Y-%m-%d-%H')
    return formatted_date

def create_retriever(question_dir):
    """
    Create a simple RAG retriever based on a set of text facts 
    """
    dir_loader = DirectoryLoader(f"{question_dir}/facts", glob="**/*.txt", loader_cls=TextLoader)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)
    docs= dir_loader.load()
    print(f"Docs#:{len(docs)}")
    chunks = text_splitter.split_documents(docs)
    chunks = filter_complex_metadata(chunks)
    vector_store = Chroma.from_documents(documents=chunks, embedding=FastEmbedEmbeddings())
    retriever = vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": 3,                 # how much rows to extract
                "score_threshold": 0.4,
            },
        )
    return retriever

def test_local_retrieval_qa(model: str, df: pd.DataFrame, retriever: any):
    """
        Core "rag" function: for the given model, run the data frame provided and return collected responses

    """
    print(f"Testing model: {model}")
    prompt = PromptTemplate.from_template(
        """
        <s> [INST] You are an assistant for question-answering tasks. Use the following pieces of retrieved context
        to answer the question. If you don't know the answer, just say that you don't know. Use three sentences
        maximum and keep the answer concise. [/INST] </s>
        [INST] Question: {question}
        Context: {context}
        Answer: [/INST]
        """
    )
    llm=ChatOllama(
            base_url="http://localhost:11434",
            model=model, temperature=0, seed=1)
    chain = ({"context": retriever, "question": RunnablePassthrough()}
                        | prompt
                        | llm
                        | StrOutputParser())
    predictions = []
    for _it, row in tqdm(df.iterrows(), total=len(df)):
        # print("Processing {} {}".format(it, row["question"]))
        # print("Context:",retriever.invoke(row["question"]))
        resp = chain.invoke(row["question"])
        print(resp)
        predictions.append(resp)
    df[f"{model}_result"] = predictions
    return df

class DataModel:
    """
    Used to store global settings
    """
    def __init__(self, df,retriever,debug_mode,question_dir,output_dir):
        self.df=df
        self.retriever=retriever
        self.debug_mode=debug_mode
        self.question_dir=question_dir
        self.output_dir=output_dir


@click.group()
@click.option('--debug/--no-debug', default=False,help="Enable lanchian debugger (to spot bugs)")
@click.option('--question-dir', default="./data", envvar="LAMINER_QUESTION_DIR", help="Data dir containing question sets and rag facts")
@click.option('--output-dir',  required=True, envvar="LAMINER_OUTPUT_DIR", help="Destination directory for rag tests")
@click.pass_context
def cli(ctx,debug, question_dir,output_dir ):
    """ 
        Examiner run.py command to run rag or collect reports.   

        Run

            run.py <command> --help 
        
        for details on every command.
        
    """
    if debug:
        set_debug(True)
    df = pd.read_csv(question_dir+"/questions/test1.csv")
    r=create_retriever(question_dir)
    ctx.obj=DataModel(df,r,debug,question_dir,output_dir)


@cli.command()
@click.pass_obj
@click.argument("models",  nargs=-1)
def rag(data_model: DataModel,models):
    """
    Run the required models using rag dataset
    Example of models: gemma:2b

    Super prod list ( "mistral-openorca:7b", "mistral:7b","llama2:7b","gemma:2b", "zephyr", "orca-ini", "phi"  )
    GG At the moment llama3 has a new prompt and does not work

    """
    formatted_date=build_file_prefix()
    dest_dir=f"{data_model.output_dir}/{formatted_date}_qa_retrieval_prediction.csv"
    print(f"Destination report:{dest_dir}")
    for current_model in models:
        test_local_retrieval_qa(current_model,data_model.df,data_model.retriever)
        data_model.df.to_csv(dest_dir, index=False)

@cli.command()
@click.pass_obj
@click.argument("qa_file",type=click.File('rb'))
@click.argument("models",  nargs=-1)
def report(data_model: DataModel, qa_file, models):
    """
    Open a qa_retrival prediction csv, and for every model compute results

    """
    prefix=build_file_prefix()
    report_file=f"{data_model.output_dir}/{prefix}-report.csv"
    print(f"Reporting into {report_file}")
    # TP: Statements that are present in both the answer and the ground truth.
    # FP: Statements present in the answer but not found in the ground truth.
    # FN: Relevant statements found in the ground truth but omitted in the answer.
    df = pd.read_csv(qa_file)
    total=len(df)
    print(f"Total questions:{total} Models to check:{len(models)}")
    tp_list=[]
    for model in models:
        precision = compute_precision_kpi(data_model, df, total, model)
        print(f"{model} Precision: {precision}")
        tp_list.append(precision)
    report_output = pd.DataFrame({
        "model": models,
        "precision": tp_list
    })
    report_output.to_csv(report_file, index=False)

def compute_precision_kpi(data_model, df, total, model):
    """
    Compute the TP := True Positive 
    """
    true_positive_score=0.0
    if data_model.debug_mode:
        print(f"Processing {model}")
        # load question,answer,gemma:2b_result
    for _i,row in df.iterrows():
        model_answer=row[f"{model}_result"]
        expected_answer=str(row["answer"])
        if data_model.debug_mode:
            print("Q:",row["question"])
            print("\tA:",model_answer)
        if expected_answer in model_answer:
            true_positive_score=true_positive_score+1
        else:
                # Try harder
            pattern=re.compile(expected_answer, re.IGNORECASE)
            if pattern.search(model_answer):
                true_positive_score=true_positive_score+1
                if data_model.debug_mode:
                    print("REGEXP MATCH")
            elif data_model.debug_mode:
                print(f"*FAILED* Expected:{expected_answer}")
    precision= true_positive_score/total
    return precision

if __name__ == '__main__':
    cli(obj={})
