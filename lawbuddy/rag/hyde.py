from dotenv import load_dotenv
from llama_index.core import (PromptTemplate, QueryBundle, Response,
                              SimpleDirectoryReader, VectorStoreIndex)
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.prompts.prompt_type import PromptType
from llama_index.readers.file import PagedCSVReader

from lawbuddy.rag.base_pipeline import BasePipeline
from lawbuddy.rag.types import QueryTransformType


class Hyde(BasePipeline):
    VECTOR_SPACE_PATH = "spaces/hyde_rag"
    HYDE_QUERY_TMPL = (
        "Please write a passage to answer the question (TH)\n"
        "Try to include as many key details as possible.\n"
        "\n"
        "\n"
        "{context_str}\n"
        "\n"
        "\n"
        'Passage:"""\n'
    )

    def __init__(self, model):
        super().__init__(model)

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

        self.retriever = self.vector_store_index.as_retriever(similarity_top_k=3)

    def load_vector_store(self, path: str = VECTOR_SPACE_PATH):
        super().load_vector_store(path)
        self.retriever = self.vector_store_index.as_retriever(similarity_top_k=3)

    def query(self, query: str, query_transform_mode: QueryTransformType = QueryTransformType.SIMPLE, verbose=False):
        if query_transform_mode == QueryTransformType.SIMPLE:
            hypo_answer = self.llm.complete(PromptTemplate(self.HYDE_QUERY_TMPL, prompt_type=PromptType.SUMMARY).format(context_str=query)).text

            retrieved_nodes = self.retriever.retrieve(hypo_answer)
            query_bundle = QueryBundle(query)
            reranked_nodes = self.reranker.postprocess_nodes(retrieved_nodes, query_bundle)

            retrieved_contents = set([n.node.get_content() for n in reranked_nodes])
            context_str = "\n\n".join(list(retrieved_contents))
            
            response = self.llm.complete(self.SYSTEM_PROMPT.format(context_str=context_str, question=query))

        elif query_transform_mode == QueryTransformType.CHUNK:
            hypo_answer = self.llm.complete(PromptTemplate(self.HYDE_QUERY_TMPL, prompt_type=PromptType.SUMMARY).format(context_str=query)).text
            nodes = self.transform_query(hypo_answer, query_transform_mode)

            retrieved_doc = []
            for sentence in nodes:
                retrieved_doc.extend(self.retriever.retrieve(sentence.text))

            query_bundle = QueryBundle(query)
            reranked_nodes = self.reranker.postprocess_nodes(retrieved_doc, query_bundle)

            retrieved_contents = set([n.node.get_content() for n in reranked_nodes])
            context_str = "\n\n".join(list(retrieved_contents))

            response = self.llm.complete(self.SYSTEM_PROMPT.format(context_str=context_str, question=query))
            
        else:
            raise ValueError(f"Unsupported query_transform_mode: {query_transform_mode}")

        if verbose:
            print("Retrieved nodes:".center(60, "-"))
            for node in reranked_nodes:
                text = node.text
                score = node.score
                print(f"Score: {score}\n{text}\n")
            print("-"*60)

        return response
