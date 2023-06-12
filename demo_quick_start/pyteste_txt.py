from dotenv import load_dotenv, find_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.embeddings import OpenAIEmbeddings
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain import VectorDBQA
# pip install Chromadb  持久化向量数据库
from langchain.document_loaders import DirectoryLoader


def create_embeddings():
    loader = DirectoryLoader('./rawdata', glob='*.txt')
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=20, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    print(texts)
    embeddings = OpenAIEmbeddings()
    docsearch = DocArrayInMemorySearch.from_documents(texts, embeddings)
    return docsearch

def run_query(vectorstore, query):
    llm = ChatOpenAI(temperature=0.0, max_tokens=1024)
    qa_stuff = VectorDBQA.from_chain_type(llm=llm, chain_type="map_reduce", vectorstore=vectorstore, return_source_documents=False)
    response = qa_stuff({"query": query})
    return response

def main():
    _ = load_dotenv(find_dotenv())

    retriever = create_embeddings()
    
    query = "What did Wukong tell the four old monkeys?(translate to chinese),summerize at 20words"
    # query = "How much change did Goku learn?(translate to chinese),summerize at 20words"
    response = run_query( retriever, query)
    print("response:!!!!!!!!!!!!!!!!!!!!!!!!!",response)

if __name__ == '__main__':
    main()