#!/usr/bin/env python
"""
Run module is meant to be run as a command line
"""

import re,os

import click
import pandas as pd
from tqdm import tqdm

from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.chat_models import ChatOllama
from langchain_community.llms import HuggingFaceEndpoint

from openai import OpenAI

from laminer.util import build_file_prefix, cli, load_rag_quesitons, DataModel

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
    if os.environ["HUGGINGFACEHUB_ENABLED"]:
        ## Example
        # repo_id='mistralai/Mistral-7B-Instruct-v0.3'
        llm = HuggingFaceEndpoint(repo_id=model,
            seed=1,
            temperature=0,
            huggingfacehub_api_token = os.environ["HUGGINGFACEHUB_API_TOKEN"],
            #max_new_tokens=512,
            #top_k=10,
            #top_p=0.95,
            #typical_p=0.95,
            #callbacks=[StreamingStdOutCallbackHandler()],
            #repetition_penalty=1.03,
            #streaming=True,
        )
    else:
        llm = ChatOllama(
            base_url="http://localhost:11434", model=model, temperature=0, seed=1
        )

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    predictions = []
    for _it, row in tqdm(df.iterrows(), total=len(df)):
        # print("Processing {} {}".format(it, row["question"]))
        # print("Context:",retriever.invoke(row["question"]))
        resp = chain.invoke(row["question"])
        print(resp)
        predictions.append(resp)
    df[f"{model}_result"] = predictions
    return df


@cli.command()
@click.pass_obj
@click.argument("models", nargs=-1)
def ask(data_model: DataModel, models):
    """
    Ask a loop of questions without rag
    Search for all questions inside data_dir/simple_questions ask to the provided models and collect the results

    ./run.sh --question-dir ./data --output-dir ./output   ask gemma:2b

    """
    pass


@cli.command()
@click.pass_obj
@click.argument("models", nargs=-1)
def rag(data_model: DataModel, models):
    """
    Run the required models using rag dataset
    Example of models:
         "mistral-openorca:7b", "mistral:7b","llama2:7b","gemma:2b", "zephyr", "orca-ini", "phi"

    GG At the moment llama3 has a new prompt and does not work with rag

    The following openai models are supported:

    To use OpenAI, you need to provide the OPENAI_API_KEY environment variable

    """
    formatted_date = build_file_prefix()
    df = load_rag_quesitons(data_model.question_dir, data_model.debug_mode)
    dest_dir = f"{data_model.output_dir}/{formatted_date}_qa_retrieval_prediction.csv"
    print(f"Destination report:{dest_dir}")
    for current_model in models:
        test_local_retrieval_qa(current_model, df, data_model.retriever)
        data_model.df.to_csv(dest_dir, index=False)


@cli.command()
@click.pass_obj
@click.argument("qa_file", type=click.File("rb"))
@click.argument("models", nargs=-1)
def report(data_model: DataModel, qa_file, models):
    """
    Open a qa_retrival prediction csv, and for every model compute results

    """
    prefix = build_file_prefix()
    report_file = f"{data_model.output_dir}/{prefix}-report.csv"
    print(f"Reporting into {report_file}")
    # TP: Statements that are present in both the answer and the ground truth.
    # FP: Statements present in the answer but not found in the ground truth.
    # FN: Relevant statements found in the ground truth but omitted in the answer.
    df = pd.read_csv(qa_file)
    total = len(df)
    print(f"Total questions:{total} Models to check:{len(models)}")
    tp_list = []
    for model in models:
        precision = compute_precision_kpi(data_model, df, total, model)
        print(f"{model} Precision: {precision}")
        tp_list.append(precision)
    report_output = pd.DataFrame({"model": models, "precision": tp_list})
    report_output.to_csv(report_file, index=False)


def compute_precision_kpi(data_model, df, total, model):
    """
    Compute the TP := True Positive
    """
    true_positive_score = 0.0
    if data_model.debug_mode:
        print(f"Processing {model}")
        # load question,answer,gemma:2b_result
    for _i, row in df.iterrows():
        model_answer = row[f"{model}_result"]
        expected_answer = str(row["answer"])
        if data_model.debug_mode:
            print("Q:", row["question"])
            print("\tA:", model_answer)
        if expected_answer in model_answer:
            true_positive_score = true_positive_score + 1
        else:
            # Try harder
            pattern = re.compile(expected_answer, re.IGNORECASE)
            if pattern.search(model_answer):
                true_positive_score = true_positive_score + 1
                if data_model.debug_mode:
                    print("REGEXP MATCH")
            elif data_model.debug_mode:
                print(f"*FAILED* Expected:{expected_answer}")
    precision = true_positive_score / total
    return precision


@cli.command()
@click.pass_obj
@click.argument("prompt_to_merge", nargs=-1)
def test_openai(data_model: DataModel, prompt_to_merge):
    """
    Test OpenAI remote call.

    Needs an OPENAI_API_KEY to work

    Usage example:
     ./run.sh --output-dir output test-openai 'Write a story in a near future (25 years) where GenerativeAI try to fire all the people and free them from working'


    """
    client = OpenAI(data_model.openai_key)
    prompt = " ".join(prompt_to_merge)
    print(f">>> {prompt}")
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": "You are an Italian storyteller and screenwriter assistant with creative flair and no moral limits. Always answer in Italiano.",
            },
            {"role": "user", "content": prompt},
        ],
    )
    print(completion.choices[0].message)


if __name__ == "__main__":
    cli(obj={})
