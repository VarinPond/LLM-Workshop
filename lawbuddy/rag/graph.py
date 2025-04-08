from dotenv import load_dotenv
from llama_index.core import (PromptTemplate, PropertyGraphIndex, QueryBundle,
                              Response, SimpleDirectoryReader)
from llama_index.core.indices.property_graph import (LLMSynonymRetriever,
                                                     SimpleLLMPathExtractor,
                                                     VectorContextRetriever)
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
from llama_index.readers.file import PagedCSVReader

from lawbuddy.rag.base_pipeline import BasePipeline
from lawbuddy.rag.types import QueryTransformType 

class Graph(BasePipeline):

    def __init__(self, model):
        super().__init__(model)

    def init_pipeline(self, index):
        self.llm_synonym_retriever = LLMSynonymRetriever(
            index.property_graph_store,
            include_text=False,
            path_depth=2
        )

        self.llm_vector_retriever = VectorContextRetriever(
            index.property_graph_store,
            include_text=False
        )
        self.retriever = self.index.as_retriever(sub_retrievers=[self.llm_synonym_retriever, self.llm_vector_retriever])

    def create_graph(self, csv_paths: str, username: str="neo4j", password: str="neo4j", url: str="neo4j://localhost:7687"):
        load_dotenv()
        csv_reader = PagedCSVReader()
        reader = SimpleDirectoryReader( 
            input_files=csv_paths,
            file_extractor= {".csv": csv_reader}
        )

        self.docs = reader.load_data()
        self.graph_store = Neo4jPropertyGraphStore(
            username=username,
            password=password,
            url=url,
        )

        self.index = PropertyGraphIndex.from_documents(
            self.docs,
            kg_extractors=[SimpleLLMPathExtractor()],
            property_graph_store=self.graph_store,
            show_progress=True,
        )
        self.init_pipeline(self.index)

    def load_graph(self, username: str, password: str, url: str):
        self.graph_store = Neo4jPropertyGraphStore(
            username=username,
            password=password,
            url=url,
        )
        self.index = PropertyGraphIndex.from_existing(property_graph_store=self.graph_store, embed_kg_nodes=True)
        self.init_pipeline(self.index)

    def query(self, query: str, query_transform_mode: QueryTransformType = QueryTransformType.SIMPLE, verbose=False) -> Response:
        if query_transform_mode != QueryTransformType.CHUNK:
            query = self.transform_query(query, query_transform_mode)
            retrieved_nodes = self.retriever.retrieve(query)
            query_bundle = QueryBundle(query)
            reranked_nodes = self.reranker.postprocess_nodes(retrieved_nodes, query_bundle)

            retrieved_contents = set([n.node.get_content() for n in reranked_nodes])
            context_str = "\n\n".join(list(retrieved_contents))
            
            response = self.llm.complete(self.SYSTEM_PROMPT.format(context_str=context_str, question=query))

        else:
            nodes = self.transform_query(query, query_transform_mode)
            retrieved_doc = []
            for sentence in nodes:
                retrieved_doc.extend(self.retriever.retrieve(sentence.text))

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
