from haystack.document_stores.in_memory import InMemoryDocumentStore
from datasets import load_dataset
from haystack import Document
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.components.builders import PromptBuilder
from haystack.components.generators import OpenAIGenerator
import os
from getpass import getpass
from haystack import Pipeline

def initialize_document_store():
    return InMemoryDocumentStore()

def load_and_prepare_documents():
    dataset = load_dataset("bilgeyucel/seven-wonders", split="train")
    docs = [Document(content=doc["content"], meta=doc["meta"]) for doc in dataset]
    return docs

def embed_documents(docs, model_name="sentence-transformers/all-MiniLM-L6-v2"):
    doc_embedder = SentenceTransformersDocumentEmbedder(model=model_name)
    doc_embedder.warm_up()
    docs_with_embeddings = doc_embedder.run(docs)
    return docs_with_embeddings

def write_documents_to_store(document_store, docs_with_embeddings):
    document_store.write_documents(docs_with_embeddings["documents"])

def initialize_text_embedder(model_name="sentence-transformers/all-MiniLM-L6-v2"):
    return SentenceTransformersTextEmbedder(model=model_name)

def initialize_retriever(document_store):
    return InMemoryEmbeddingRetriever(document_store)

def initialize_prompt_builder():
    template = """
    Given the following information, answer the question.

    Context:
    {% for document in documents %}
        {{ document.content }}
    {% endfor %}

    Question: {{question}}
    Answer:
    """
    return PromptBuilder(template=template)

def initialize_generator():
    if "OPENAI_API_KEY" not in os.environ:
        os.environ["OPENAI_API_KEY"] = getpass("Enter OpenAI API key:")
    return OpenAIGenerator(model="gpt-3.5-turbo")

def create_pipeline(text_embedder, retriever, prompt_builder, generator):
    pipeline = Pipeline()
    pipeline.add_component("text_embedder", text_embedder)
    pipeline.add_component("retriever", retriever)
    pipeline.add_component("prompt_builder", prompt_builder)
    pipeline.add_component("llm", generator)
    pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
    pipeline.connect("retriever", "prompt_builder.documents")
    pipeline.connect("prompt_builder", "llm")
    return pipeline

def main():
    document_store = initialize_document_store()
    docs = load_and_prepare_documents()
    docs_with_embeddings = embed_documents(docs)
    write_documents_to_store(document_store, docs_with_embeddings)
    
    text_embedder = initialize_text_embedder()
    retriever = initialize_retriever(document_store)
    prompt_builder = initialize_prompt_builder()
    generator = initialize_generator()
    
    pipeline = create_pipeline(text_embedder, retriever, prompt_builder, generator)
    
    while True:
        question = input("Ask your question: ").strip()
        if question.lower() in ["e", "exit"]:
            print("Exiting the program.")
            break
        
        response = pipeline.run({"text_embedder": {"text": question}, "prompt_builder": {"question": question}})
        print("Answer:", response["llm"]["replies"][0])

if __name__ == "__main__":
    main()
