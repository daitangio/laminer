# laminer
lama-examiner simple code to test GenAI Large Language model (Llama) precision based on a corpus of tests

# Getting started
Create a virtualenv and install poetry

    virtualenv venv
    source venv/bin/activate
    pip install poetry==1.2.0

Then use poetry to install examiner dependencies

Suppose you have your ollama in a host calld fatlama, proxy it with

    ssh -L 11434:127.0.0.1:11434 -N -f fatlama