# Laminer
lama-examiner (Laminer) is a simple code to test GenAI Large Language model precision based on a corpus of tests.

It can test rag sets on a provided dataset.


[![JUnit_Pylint](https://github.com/daitangio/laminer/actions/workflows/unittest.yml/badge.svg)](https://github.com/daitangio/laminer/actions/workflows/unittest.yml)

# Getting started
Create a virtualenv and install poetry:

    virtualenv venv && source venv/bin/activate
    pip install poetry==1.2.0
    poetry install --with dev

After, you need an ollama instance. You can use the runOllamaServer.sh script to run it via docker.
Otherwise you can use a proxed instance

## Proxed Ollama
Suppose you have your ollama in a host calld fatlama, proxy it with

    ssh -L 11434:127.0.0.1:11434 -N -f fatlama


# Command examples

To run some tests and then collect results try

    ./run.sh --question-dir ./data --output-dir ./output    rag gemma:2b mistral:7b
    ./run.sh --question-dir ./data --output-dir ./output report output/2024-05-08_qa_retrieval_prediction.csv gemma:2b

# How to configure expected answer (WIP)

Expected answer can be regexp


## TODOs

[ ] Implement new 'simple' command for non-RAG flows (simple QA)
    based on small cobol translations

[ ] Provide more usage examples, and tutorial based on langchain:
    https://python.langchain.com/v0.1/docs/get_started/quickstart/

[ ] Review MMMLU https://github.com/hendrycks/test?tab=readme-ov-file#measuring-massive-multitask-language-understanding 
    and consider integrating it
    with a specific command
[ ] Add contribution.rst and issue template

[ ] Include a test example based on phi3


# Major references

1) https://medium.com/@vndee.huynh/how-to-effectively-evaluate-your-rag-llm-applications-2139e2d2c7a4
2) Read about poetry package managere here: https://python-poetry.org/docs/
3) Tox https://tox.wiki/en/4.15.0/