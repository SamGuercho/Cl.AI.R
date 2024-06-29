from abc import ABC, abstractmethod
import os, json
from pathlib import Path

import uuid
from typing import List, Dict, Optional
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.storage.docstore import SimpleDocumentStore, BaseDocumentStore
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core import VectorStoreIndex, ServiceContext, SimpleDirectoryReader, GPTVectorStoreIndex
from llama_index.core.embeddings import BaseEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.vector_stores.types import BasePydanticVectorStore, VectorStore
import chromadb
from llama_index.core import StorageContext
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.schema import NodeWithScore, BaseNode, TextNode, Document

from dotenv import load_dotenv
from overrides import overrides
import pinecone

load_dotenv()


class KnowledgeBase(ABC):
    """Knowledge Base class is an index with an embedding model to search for nodes in the knowledge base."""

    def __init__(self, embedding_model: BaseEmbedding, **kwargs):
        # TODO connect an index to the knowledgeBase

        self.embedding_model = embedding_model
        self._storage_context = None
        self.index: VectorStoreIndex | None = None

    def search(self, query: str, top_k: int = 5) -> List[NodeWithScore]:
        """Search the knowledge base for the most relevant nodes to the query."""
        # TODO add here the MetadataFilters and a constant with the excluded fields to not take into account in the search
        query_engine = self.index.as_query_engine(
            similarity_top_k=top_k
        )
        return query_engine.retrieve(query)

    def index(self, documents: Document) -> None:
        # TODO index the node to the knowledgebase
        return None

class KBFactory():
    @staticmethod
    def get_kb(kb_name:str, kb_type:str, embedding_model: BaseEmbedding, **kwargs) -> KnowledgeBase:
        if kb_type == "chroma":
            return ChromaKB(**kwargs)
        elif kb_type == "pinecone":
            return PinceConeKB(index_name=kb_name, embedding_model=embedding_model, **kwargs)
        else:
            raise ValueError(f"Knowledge Base type [{kb_type}] not supported.")
class ChromaKB(KnowledgeBase):
    """Knowledge Base class using a chromadb vector store"""
    _path = os.environ['CHROMA_DB_PATH']
    _index_dir = "index"
    _docstore_dir = 'index_store.json'
    def __init__(self, kb_path: str, collection_name: str, embedding_model: BaseEmbedding, vector_store: VectorStore, **kwargs):
        super().__init__(embedding_model, **kwargs)
        self._kb_path = Path(kb_path)
        # TODO add the load_db method
        self._service_context = ServiceContext.from_defaults(
            embed_model=self.embedding_model
        )
        self._storage_context = StorageContext.from_defaults(
            vector_store=vector_store,
            docstore=SimpleDocumentStore(),
            # persist_dir=str(Path(self._path) / collection_name/ self._kb_path)
        )
        self._docstore: BaseDocumentStore = None

    @classmethod
    def create_chroma_kb(cls,
                         kb_path: str,
                         collection_name: str,
                         embedding_model_name: str | None = None,
                         **embed_model_kwargs
                         ):
        """Create a ChromaKnowledgeBase instance."""
        #TODO create also the load method for existing chromadb vector store
        if embedding_model_name is None:
            embedding_model_name = os.environ['DEFAULT_EMBEDDING_MODEL']
        embedding_model = HuggingFaceEmbedding(embedding_model_name, **embed_model_kwargs)
        vector_store = cls._load_chroma_db_vector_store(cls, collection_name=kb_path)
        cls._create_config_json_file(cls, kb_path=kb_path, embedding_model=embedding_model)
        return cls(
            kb_path=kb_path,
            collection_name=collection_name,
            embedding_model=embedding_model,
            vector_store=vector_store
        )

    def index_docs(self, documents: List[Document]) -> None:
        """Index the documents to the knowledge base."""
        self.index = VectorStoreIndex.from_documents(
            documents=documents,
            storage_context=self._storage_context,
            embed_model=self.embedding_model
        )

    def get_text_labels(self, texts: List[str]) -> List[str]:
        """Get the labels for the texts."""
        return [""] * len(texts)

    def _load_chroma_db_vector_store(self, collection_name: str) -> ChromaVectorStore:
        """Load the chroma db vector store."""
        chroma_client = chromadb.PersistentClient(
            path=str(Path(self._path) / collection_name/ self._index_dir)
        )
        try:
            chroma_collection = chroma_client.create_collection(collection_name)
        except Exception as e:
            #TODO add a meaning warning to explain that we need to load and not to create
            print("If the db already exists, you need to load instead of creating.")
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        return vector_store

    def _create_config_json_file(self, kb_path: str, embedding_model: BaseEmbedding):
        config = {
            "embedding_model_name": embedding_model.model_name,
            "max_length": embedding_model.max_length
        }
        with open(Path(self._path) / kb_path / "config.json", "w") as f:
            json.dump(config, f)

class PinceConeKB(KnowledgeBase):
    """Knowledge Base class using a Pinecone vector store"""
    _default_text_key = "text"
    def __init__(self, index_name: str, embedding_model: BaseEmbedding, **kwargs):
        super().__init__(embedding_model, **kwargs)
        self._client = pinecone.Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
        self._service_context = ServiceContext.from_defaults(embed_model=self.embedding_model)
        self.index = self._client.Index(index_name)
        pc_vstore : PineconeVectorStore = PineconeVectorStore(
            pinecone_index=self.index,
            text_key=self._default_text_key
        )
        self._vector_store: VectorStoreIndex =  VectorStoreIndex.from_vector_store(pc_vstore)
        self._storage_context = StorageContext.from_defaults(
            vector_store=self._vector_store
        )

    def _get_or_build_index(self, index_name: str) -> pinecone.Pinecone:
        #TODO not used today, need to be integrated
        try:
            indexes = pinecone.list_indexes()
        except Exception as e:
            return None
        if index_name in indexes:
            return pinecone.Pinecone(index_name)
        else:
            index = pinecone.create_index(
                index_name, dimension=1536, metric="euclidean", pod_type="p1"
            )
            return pinecone.Index(index_name)
    def index_docs(self, documents: List[Document]|None = None) -> None:
        self.index = GPTVectorStoreIndex.from_documents(
            documents,
            storage_context=self._storage_context,
            service_context=self._service_context
        )

    def add_docs(self, documents: List[Document]) -> None:
        entries = []
        for doc in documents:
            vector = self.embedding_model.get_text_embedding(doc.text)
            entries.append({
                "id":str(uuid.uuid4()),
                "values": vector,
                "metadata": doc.metadata
            })
            self.index.upsert(entries)
            self.index.delete()

    @overrides
    def search(self, query: str, top_k: int = 5) -> List[NodeWithScore]:
        retriever = self._vector_store.as_retriever(
            similarity_top_k=top_k
        )
        return retriever.retrieve(query)


if __name__ == "__main__":
    embedding_model = OpenAIEmbedding(model='text-embedding-ada-002')
    pc_kb = PinceConeKB(index_name='neogenai', embedding_model=embedding_model)
    with open('../data/quicktest_db.json', 'r') as file:
        data = json.load(file)
    documents = []

    for k, v in data.items():
        print(k, v)
        documents.append({
            'id': k,
            'text': v["text"]
        })
    documents[0]['text'] = "modified_doc" + documents[0]['text']
    llama_documents = [Document(text=doc['text'], metadata=doc) for doc in documents]
    # pc_kb.index(llama_documents)
    pc_kb.search("What is the meaning of life?")
    print()
