from dotenv import load_dotenv, find_dotenv
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import CSVLoader
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.embeddings import OpenAIEmbeddings
from langchain.embeddings.openai import OpenAIEmbeddings


#env 文件写法 export OPENAI_API_KEY='key'
def load_csv(file_path):
    loader = CSVLoader(file_path=file_path)
    docs = loader.load()
    return docs

def create_embeddings(docs):
    embeddings = OpenAIEmbeddings()
    db = DocArrayInMemorySearch.from_documents(
        docs, 
        embeddings
    )
    retriever = db.as_retriever()
    return retriever

def run_query(llm, retriever, query):
    qa_stuff = RetrievalQA.from_chain_type(llm=llm, chain_type="refine", retriever=retriever, verbose=True)
    response = qa_stuff.run(query)
    return response

def main():
    _ = load_dotenv(find_dotenv())
    file_path = 'OutdoorClothingCatalog_1000.csv'
    docs = load_csv(file_path)
    retriever = create_embeddings(docs)
    llm = ChatOpenAI(temperature=0.0, max_tokens=1024)
    query ="Please list all your shirts with sun protection \
    in a table in markdown and summarize each one."
    response = run_query(llm, retriever, query)
    print(response)

if __name__ == '__main__':
    main()