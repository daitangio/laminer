import pandas as pd
from tqdm import tqdm
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOllama
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

def create_retriever():
    file_loader = TextLoader("./data/fact1.txt")
    text_splitter = CharacterTextSplitter(chunk_size=1000)
    chunks = text_splitter.split_documents(file_loader.load())
    vector_store = FAISS.from_documents(chunks, HuggingFaceEmbeddings())
    retriever = vector_store.as_retriever()
    return retriever

def test_local_retrieval_qa(model: str, df: pd.DataFrame, retriever: FAISS):
    print("Testing model: {}".format(model))
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
    for it, row in tqdm(df.iterrows(), total=len(df)):
        # print("Processing {} {}".format(it, row["question"]))
        resp = chain.invoke(row["question"])
        print(resp)
        predictions.append(resp)
    
    df[f"{model}_result"] = predictions
    


if __name__ == "__main__":
    df = pd.read_csv("./data/test1.csv")
    r=create_retriever()
    # df = pd.DataFrame({
    #     "question": questions,
    #     "answer": answers
    # })    
    test_local_retrieval_qa("gemma:2b",df,r)
    # test_local_retrieval_qa("mistral:7b",df,r)
    # test_local_retrieval_qa("llama2")
    # test_local_retrieval_qa("zephyr")
    # test_local_retrieval_qa("orca-mini")
    # test_local_retrieval_qa("phi")
    df.to_csv("./output/qa_retrieval_prediction.csv", index=False)