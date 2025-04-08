from pathlib import Path

from dotenv import load_dotenv
from llama_index.core import (PromptTemplate, QueryBundle, Response,
                              SimpleDirectoryReader, SimpleKeywordTableIndex,
                              StorageContext, VectorStoreIndex,
                              load_index_from_storage)
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.retrievers import (BaseRetriever,
                                         KeywordTableSimpleRetriever,
                                         VectorIndexRetriever)
from llama_index.readers.file import PagedCSVReader
from lawbuddy.rag.types import QueryTransformType
from lawbuddy.rag.base_pipeline import BasePipeline


class HybridRetriever(BaseRetriever):
    """Custom retriever that performs both semantic search and hybrid search."""

    def __init__(
        self,
        vector_retriever: VectorIndexRetriever,
        keyword_retriever: KeywordTableSimpleRetriever,
        mode: str = "AND",
    ) -> None:
        """Init params."""

        self._vector_retriever = vector_retriever
        self._keyword_retriever = keyword_retriever
        if mode not in ("AND", "OR"):
            raise ValueError("Invalid mode.")
        self._mode = mode
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle):
        """Retrieve nodes given query."""

        vector_nodes = self._vector_retriever.retrieve(query_bundle)
        keyword_nodes = self._keyword_retriever.retrieve(query_bundle)

        vector_ids = {n.node.node_id for n in vector_nodes}
        keyword_ids = {n.node.node_id for n in keyword_nodes}

        combined_dict = {n.node.node_id: n for n in vector_nodes}
        combined_dict.update({n.node.node_id: n for n in keyword_nodes})

        if self._mode == "AND":
            retrieve_ids = vector_ids.intersection(keyword_ids)
        else:
            retrieve_ids = vector_ids.union(keyword_ids)

        retrieve_nodes = [combined_dict[rid] for rid in retrieve_ids]
        print(f"Retrieved {len(retrieve_nodes)} nodes.")
        return retrieve_nodes

class Hybrid(BasePipeline):
    VECTOR_SPACE_PATH = "spaces/hybrid_rag"
    DEFAULT_QUERY_KEYWORD_EXTRACT_TEMPLATE_TMPL = (
        "A question is provided below. Given the question, extract up to {max_keywords} "
        "keywords from the text. Focus on extracting the keywords that we can use "
        "to best lookup answers to the question. Avoid stopwords.\n"
        "The keywords should be (TH language only) law content ex. ทำร้ายร่างกาย, คดีอาญา, การสอบสวน, โดนโกง, ลูกจ้าง.\n"
        "---------------------\n"
        "{question}\n"
        "---------------------\n"
        "Provide keywords in the following comma-separated format: 'KEYWORDS: <keywords>'\n"
    )
    DEFAULT_QUERY_KEYWORD_EXTRACT_TEMPLATE = PromptTemplate(DEFAULT_QUERY_KEYWORD_EXTRACT_TEMPLATE_TMPL)

    def __init__(self, model):
        super().__init__(model)

    @staticmethod
    def completion_to_prompt(completion):
        return f'<|im_start|>system\n{BasePipeline.SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n{completion}<|im_end|>\n<|im_start|>assistant\n'

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
        self.vector_store_index.storage_context.persist(str(Path(save_dir) / "vector_store"))
        self.keyword_index = SimpleKeywordTableIndex(self.nodes, keyword_extract_template=self.DEFAULT_QUERY_KEYWORD_EXTRACT_TEMPLATE)
        self.keyword_index.storage_context.persist(str(Path(save_dir) / "keyword_index"))

        self.vector_retriever = VectorIndexRetriever(index=self.vector_store_index, similarity_top_k=10)
        self.keyword_retriever = KeywordTableSimpleRetriever(index=self.keyword_index)
        self.custom_retriever = HybridRetriever(self.vector_retriever, self.keyword_retriever, mode="OR")

    def load_vector_store(self, path: str = VECTOR_SPACE_PATH):
        self.vector_store_index = load_index_from_storage(StorageContext.from_defaults(persist_dir=str(Path(path) / "vector_store")))
        self.keyword_index = load_index_from_storage(StorageContext.from_defaults(persist_dir=str(Path(path) / "keyword_index")))

        self.vector_retriever = VectorIndexRetriever(index=self.vector_store_index, similarity_top_k=10)
        self.keyword_retriever = KeywordTableSimpleRetriever(index=self.keyword_index)
        self.custom_retriever = HybridRetriever(self.vector_retriever, self.keyword_retriever)


    def query(self, query: str, query_transform_mode: QueryTransformType = QueryTransformType.SIMPLE, verbose=False) -> Response:
        if query_transform_mode != QueryTransformType.CHUNK:
            query = self.transform_query(query, query_transform_mode)
            query_bundle = QueryBundle(query)
            retrieved_nodes = self.custom_retriever.retrieve(query_bundle)
            # rerank nodes
            reranked_nodes = self.reranker.postprocess_nodes(retrieved_nodes, query_bundle)
            retrieved_contents = set([n.node.get_content() for n in reranked_nodes])
            context_str = "\n\n".join(list(retrieved_contents))

            response = self.llm.complete(self.SYSTEM_PROMPT.format(context_str=context_str, question=query))
    
        else:
            nodes = self.transform_query(query, query_transform_mode)
            retrieved_doc = []
            for sentence in nodes:
                retrieved_doc.extend(self.custom_retriever.retrieve(sentence.text))

            query_bundle = QueryBundle(query)
            reranked_nodes = self.reranker.postprocess_nodes(retrieved_doc, query_bundle)

            retrieved_contents = set([n.node.get_content() for n in reranked_nodes])
            context_str = "\n\n".join(list(retrieved_contents))

            response = self.llm.complete(self.SYSTEM_PROMPT.format(context_str=context_str, question=query))
        
        if verbose:
            print("Retrieved nodes:".center(60, "-"))
            for node in reranked_nodes:
                text = node.text
                score = node.score
                print(f"Score: {score}\n{text}\n")
            print("-"*60)

        return response

