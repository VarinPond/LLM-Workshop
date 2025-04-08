from dotenv import load_dotenv
from llama_index.core import (PromptTemplate, QueryBundle,
                              SimpleDirectoryReader, VectorStoreIndex)
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.readers.file import PagedCSVReader

from lawbuddy.rag.base_pipeline import BasePipeline
from lawbuddy.rag.types import QueryTransformType

class NoRagPipeline(BasePipeline):
    VECTOR_SPACE_PATH = "spaces/simple_rag"

    def __init__(self, model):
        super().__init__(model)

    @staticmethod
    def completion_to_prompt(completion):
        return f'<|im_start|>system\n{BasePipeline.SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n{completion}<|im_end|>\n<|im_start|>assistant\n'

    # def __init__(self, model_name: str ="openthaigpt/openthaigpt1.5-7b-instruct" , quantization_config=None, **kwargs):
    #     super().__init__(model_name, quantization_config, **kwargs)

    def create_vector_store(self, csv_paths: str, save_dir: str = VECTOR_SPACE_PATH):
        load_dotenv()
        csv_reader = PagedCSVReader()
        reader = SimpleDirectoryReader( 
            input_files=csv_paths,
            file_extractor= {".csv": csv_reader}
        )

        self.docs = reader.load_data()
        self.pipeline = IngestionPipeline()
        self.nodes = self.pipeline.run(documents=self.docs, show_progress=True)
        self.vector_store_index = VectorStoreIndex(self.nodes)
        self.vector_store_index.storage_context.persist(save_dir)

        self.retriever = self.vector_store_index.as_retriever(similarity_top_k=10, llm=self.llm)

    def load_vector_store(self, path: str = VECTOR_SPACE_PATH):
        super().load_vector_store(path)
        self.retriever = self.vector_store_index.as_retriever(similarity_top_k=10, llm=self.llm)

    def query(self, query: str, query_transform_mode: QueryTransformType = QueryTransformType.SIMPLE, verbose: bool = False):
        if query_transform_mode != QueryTransformType.CHUNK:
            query = self.transform_query(query, query_transform_mode)
            retrieved_nodes = self.retriever.retrieve(query)
            query_bundle = QueryBundle(query)
            reranked_nodes = self.reranker.postprocess_nodes(retrieved_nodes, query_bundle)

            retrieved_contents = set([n.node.get_content() for n in reranked_nodes])
            context_str = "\n\n".join(list(retrieved_contents))
            
            response = self.llm.complete(self.SYSTEM_PROMPT.format(context_str="", question=query))

        else:
            nodes = self.transform_query(query, query_transform_mode)
            retrieved_doc = []
            for sentence in nodes:
                retrieved_doc.extend(self.retriever.retrieve(sentence.text))

            query_bundle = QueryBundle(query)
            reranked_nodes = self.reranker.postprocess_nodes(retrieved_doc, query_bundle)

            retrieved_contents = set([n.node.get_content() for n in reranked_nodes])
            context_str = "\n\n".join(list(retrieved_contents))

            response = self.llm.complete(self.SYSTEM_PROMPT.format(context_str="", question=query))

        if verbose:
            print("Retrieved nodes:".center(60, "-"))
            for node in reranked_nodes:
                text = node.text
                score = node.score
                print(f"Score: {score}\n{text}\n")
            print("-"*60)

        return response
