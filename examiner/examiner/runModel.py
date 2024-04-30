import pandas as pd
from tqdm import tqdm
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOllama
from langchain.embeddings import HuggingFaceEmbeddings, FastEmbedEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain.vectorstores.utils import filter_complex_metadata

def create_retriever():
    file_loader = TextLoader("./data/fact1.txt")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)
    chunks = text_splitter.split_documents(file_loader.load())
    chunks = filter_complex_metadata(chunks)

    vector_store = Chroma.from_documents(documents=chunks, embedding=FastEmbedEmbeddings())
    retriever = vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": 3,
                "score_threshold": 0.5,
            },            
        )
    return retriever

def test_local_retrieval_qa(model: str, df: pd.DataFrame, retriever: any):
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
        print("Context:",retriever.invoke(row["question"]))
        resp = chain.invoke(row["question"])
        print(resp)
        predictions.append(resp)
    
    df[f"{model}_result"] = predictions
    


if __name__ == "__main__":
    df = pd.read_csv("./data/test1.csv")
    r=create_retriever()
    # Super prod list ( "mistral-openorca:7b", "mistral:7b","llama2:7b","gemma:2b", "zephyr", "orca-ini", "phi"  )
    # GG At the moment llama3 presnet a bug in the outuput, it seems unable to understand the prompt.
    for modelName in ( "gemma:2b", "mistral:7b"):
        test_local_retrieval_qa(modelName,df,r)    
        df.to_csv("./output/qa_retrieval_prediction.csv", index=False)        