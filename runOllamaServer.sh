#!/bin/bash
# docker run -d --gpus=all -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama:latest
docker run --rm -d -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama:latest
docker exec -it ollama ollama pull gemma:2b
docker exec -it ollama ollama pull llama3:8b
docker exec -it ollama ollama pull mistral:7b
docker exec -it ollama ollama pull mistral-openorca:7b
