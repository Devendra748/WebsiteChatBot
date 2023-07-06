# Website ChatBot

This repository contains code for utilizing the LangChain library to perform various natural language processing tasks, including text splitting, document embedding, question answering, and more.

## Installation

To install the required dependencies, please run the following commands:

```python
!pip install langchain
!pip install faiss-cpu
!pip install openai
!pip install unstructured
```

Make sure you have the necessary permissions to install packages in your environment.

## Usage

1. Set OpenAI API Key: Before using the LangChain library, you need to set your OpenAI API key as an environment variable. This can be done by adding the following line of code:

```python
import os
os.environ["OPENAI_API_KEY"] = "OPENAI_API_KEY"
```

Replace `"OPENAI_API_KEY"` with your actual OpenAI API key.

2. Document Loading: LangChain provides a `UnstructuredURLLoader` class for loading documents from URLs. You can use it as follows:

```python
from langchain.document_loaders import UnstructuredURLLoader

urls = [
    'https://www.mosaicml.com/blog/mpt-7b',
    'https://www.mosaicml.com/blog/mpt-30b',
    'https://www.mosaicml.com/blog/inference-launch'
]

loaders = UnstructuredURLLoader(urls=urls)
data = loaders.load()
```

3. Text Splitting: To split the loaded documents into smaller chunks, you can use the `CharacterTextSplitter` class from LangChain. Here's an example:

```python
from langchain.text_splitter import CharacterTextSplitter

text_splitter = CharacterTextSplitter(separator='\n',
                                      chunk_size=1000,
                                      chunk_overlap=200)

docs = text_splitter.split_documents(data)
```

4. Document Embedding: LangChain supports document embedding using OpenAI's language model. You can embed the documents using the following code:

```python
import pickle
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()
vectorStore_openAI = FAISS.from_documents(docs, embeddings)

# Save the vector store to a file
with open("faiss_store_openai.pkl", "wb") as f:
    pickle.dump(vectorStore_openAI, f)

# Load the vector store from a file
with open("faiss_store_openai.pkl", "rb") as f:
    VectorStore = pickle.load(f)
```

5. Question Answering: LangChain provides functionality for performing question answering on the embedded documents. Here's an example:

```python
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.question_answering import load_qa_chain
from langchain import OpenAI

llm = OpenAI(temperature=0)
chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=VectorStore.as_retriever())

output = chain({"question": "Is it ready for the enterprise?"}, return_only_outputs=True)
```

## Additional Resources

For more information and examples, please refer to the official documentation of the LangChain library.

## License

This project is licensed under the [MIT License](LICENSE).
