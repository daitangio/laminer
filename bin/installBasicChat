#!/bin/bash
# llama3:8b mistral:7b mistral-openorca:7b
for m in  gemma:2b phi3  ; do
    echo Pulling $m
    docker exec -it ollama ollama pull $m
done
