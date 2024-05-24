#!/bin/bash
for m in  gemma:2b phi3 llama3:8b mistral:7b mistral-openorca:7b ; do
    echo Pulling $m
    docker exec -it ollama ollama pull $m
done
