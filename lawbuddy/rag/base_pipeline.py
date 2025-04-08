import torch
from dotenv import load_dotenv
from llama_index.core import (Response, PromptTemplate, SimpleDirectoryReader, StorageContext,
                              VectorStoreIndex, load_index_from_storage, Document)
from llama_index.core.postprocessor import LLMRerank
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.llms.openai import OpenAI
from llama_index.readers.file import PagedCSVReader
from transformers import BitsAndBytesConfig
from lawbuddy.rag.types import QueryTransformType
from llama_index.llms.openai_like import OpenAILike

from typing import Union

class BasePipeline:
    SYSTEM_PROMPT = PromptTemplate(
        "นี่คือเนื้อหา\n"
        "---------------------\n"
        "{context_str}\n"
        "---------------------\n"
        "จงใช้เนื้อหาต่อไปนี้ในการตอบคำถาม พร้อมอ้างอิง `มาตรา`\n"
        "Question: {question}\n"
        "Answer: "
    )
    @staticmethod
    def messages_to_prompt(messages):
        raise NotImplementedError
    
    @staticmethod
    def completion_to_prompt(completion):
        return f'<|im_start|>system\n{BasePipeline.SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n{completion}<|im_end|>\n<|im_start|>assistant\n'

    def __init__(self, model):
        self.llm = model
        self.set_reranker(LLMRerank(choice_batch_size=5, top_n=5))

    @classmethod
    def from_local_model(cls, model_name: str, model, quantization_config=None, **kwargs):
        if quantization_config is None:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        llm = HuggingFaceLLM(
            model_name=model_name,
            tokenizer_name=model_name,
            model=model,
            # context_window=3900,
            max_new_tokens=2048,
            model_kwargs={"quantization_config": quantization_config},
            # generate_kwargs={"temperature": 0.7, "top_k": 50, "top_p": 0.95},
            messages_to_prompt=BasePipeline.messages_to_prompt,
            completion_to_prompt=BasePipeline.completion_to_prompt,
            device_map="auto",
        )
        return cls(model=llm, **kwargs)

    @classmethod
    def from_openai_model(cls, **kwargs):
        return cls(model=OpenAI(**kwargs))
    
    @classmethod
    def from_api(cls, **kwargs):
        return cls(model=OpenAILike(**kwargs)) 

    def set_reranker(self, reranker):
        self.reranker = reranker

    def create_vector_store(self, csv_paths: str, save_dir: str = "./index"):
        load_dotenv()
        csv_reader = PagedCSVReader()
        reader = SimpleDirectoryReader( 
            input_files=csv_paths,
            file_extractor= {".csv": csv_reader}
        )

        self.docs = reader.load_data()
        self.vector_store_index = VectorStoreIndex.from_documents(self.docs)
        self.vector_store_index.storage_context.persist(save_dir)

        self.query_engine = self.vector_store_index.as_query_engine(similarity_top_k=3, llm=self.llm)

    def load_vector_store(self, path: str = "./index"):
        storage_context = StorageContext.from_defaults(persist_dir=path)
        self.vector_store_index = load_index_from_storage(storage_context)

        self.query_engine = self.vector_store_index.as_query_engine(similarity_top_k=3, llm=self.llm)

    def query(self, query: str) -> Response:
        streaming_response = self.query_engine.query(query)
        return streaming_response

    def transform_query(self, query: str, query_transform_mode: QueryTransformType) -> Union[str, list]:
        if query_transform_mode == QueryTransformType.SIMPLE:
            return query
        elif query_transform_mode == QueryTransformType.CHUNK:
            nodes = SentenceSplitter(chunk_size=512, chunk_overlap=128).get_nodes_from_documents([Document(text=query)])
            return nodes
        

