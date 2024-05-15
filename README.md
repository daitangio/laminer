# Laminer
lama-examiner (Laminer) is a simple code to test GenAI Large Language model precision based on a corpus of tests.

It can test

[![Pylint](https://github.com/daitangio/laminer/actions/workflows/pylint.yml/badge.svg)](https://github.com/daitangio/laminer/actions/workflows/pylint.yml)

# Getting started
Create a virtualenv and install poetry:

    virtualenv venv
    source venv/bin/activate
    pip install poetry==1.2.0
    poetry install --with dev

After, you need an ollama instance. You can use the runOllamaServer.sh script to run it via docker.

## Proxed Ollama
Suppose you have your ollama in a host calld fatlama, proxy it with

    ssh -L 11434:127.0.0.1:11434 -N -f fatlama


# Command examples

To run some tests and then collect results try

    ./laminer/run.py --question-dir ./data --output-dir ./output  rag gemma:2b mistral:7b     
    ./laminer/run.py --question-dir ./data --output-dir ./output report output/2024-05-08_qa_retrieval_prediction.csv gemma:2b

# How to configure expected answer (WIP)

Expected answer can be regexp


# Major references

1) https://medium.com/@vndee.huynh/how-to-effectively-evaluate-your-rag-llm-applications-2139e2d2c7a4
2) Read about poetry package managere here: https://python-poetry.org/docs/
3) Tox https://tox.wiki/en/4.15.0/