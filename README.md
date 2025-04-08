# LawBuddy 🤖⚖️
A powerful Thai legal assistant with RAG techniques.

🚀 Installation
- Create an .env file with the following content:
    ```bash
    OPENAI_API_KEY=YOUR_OPENAI_API_KEY
    ```
- Install the package using pip:
    ```bash
    
    git clone https://github.com/BetterACS/LawBuddy
    cd LawBuddy
    pip install -r requirements.txt
    pip install -e .
    ```

💡 Quick Start
Using OpenAI Model
```python
from lawbuddy.rag import SimpleRagPipeline, Hybrid, Graph, Hyde, QueryTransformType
pipeline = SimpleRagPipeline.from_openai_model(model="gpt-3.5-turbo")
pipeline.create_vector_store(
    csv_paths=["laws.csv"],
    save_dir="spaces/hybrid_rag"
)

query = "โดนโกง 300 ล้านบาทไทย แต่คนโกงไม่โดนฟ้องควรทำยังไง"
response = pipeline.query(query, verbose=True)
```

📚 Vector Store Management
- Creating a New Vector Store
    ```python
    # Create vector store from CSV files
    pipeline.create_vector_store(
        csv_paths=["laws.csv"],
        save_dir="spaces/hybrid_rag"
    )
    ```

- Loading Existing Vector Store
    ```python
    pipeline.load_vector_store(path="spaces/hybrid_rag")
    ```

🚌 Query Transforms
- Simple Query Transform (default)
    ```python
    from lawbuddy.rag import QueryTransformType
    pipeline.query(
        query="โดนโกง 300 ล้านบาทไทย แต่คนโกงไม่โดนฟ้องควรทำยังไง",
        query_transform_mode=QueryTransformType.SIMPLE
    )
    ```
    No query transformation is applied.

- Chunk Query Transform
    ```python
    from lawbuddy.rag import QueryTransformType
    pipeline.query(
        query="โดนโกง 300 ล้านบาทไทย แต่คนโกงไม่โดนฟ้องควรทำยังไง",
        query_transform_mode=QueryTransformType.CHUNK
    )
    ```
    Chunk the query into smaller parts for better processing.
    Each chunk is being retrieved separately from the vector store and then concatenated later.

🤖 Load model
- Local model
    ```python
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from lawbuddy.rag import Hybrid
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained("openthaigpt/openthaigpt1.5-7b-instruct")
    tokenizer = AutoTokenizer.from_pretrained("openthaigpt/openthaigpt1.5-7b-instruct")

    # Load specialized legal adapter
    model.load_adapter("betteracs/lawbuddy-7b")

    # Initialize pipeline with local model
    pipeline = Hybrid.from_local_model(
        model_name="openthaigpt/openthaigpt1.5-7b-instruct",
        model=model
    )
    ```
- API
    ```python
    from lawbuddy.rag import SimpleRagPipeline
    pipeline = SimpleRagPipeline.from_api(
        model="typhoon-v1.5-instruct",
        api_base="https://api.opentyphoon.ai/v1",
        context_window=8192,
        is_chat_model=True,
        max_tokens=768,
        is_function_calling_model=False,
        api_key="...."
    )
    ```

💹 Graph RAG
For the Graph RAG model, you need to install the Neo4j database and run the following commands:

- Installation
    ```bash
    pip install neo4j
    pip install llama-index-vector-stores-neo4jvector
    ```
- Start Neo4j database with docker
    ```bash
    docker run \
        -p 7474:7474 -p 7687:7687 \
        -v $PWD/data:/data -v $PWD/plugins:/plugins \
        --name neo4j-apoc \
        -e NEO4J_apoc_export_file_enabled=true \
        -e NEO4J_apoc_import_file_enabled=true \
        -e NEO4J_apoc_import_file_use__neo4j__config=true \
        -e NEO4JLABS_PLUGINS=\[\"apoc\"\] \
        neo4j:latest
    ```
- Authenticate with the default username and password (neo4j/neo4j) or changing the password in browser at http://localhost:7474

- Create a new graph
    ```python
    from lawbuddy.rag import Graph
    import nest_asyncio
    nest_asyncio.apply() # Required for preventing asyncio conflicts
    
    graph = Graph.from_openai_model(model="gpt-3.5-turbo")
    graph.create_graph(
        csv_paths=["laws.csv"],
        url="neo4j://localhost:7687",
        username="neo4j", # default username
        password="neo4j" # default password
    )
    # graph.load_graph(
    #     url="neo4j://localhost:7687",
    #     username="neo4j",
    #     password="neo4j"
    # )
    ```
    This might take a while to create the graph.

🧪 Evaluation
------------------

To evaluate the model performance on specific tasks or legal document types, use the following script. This example shows how to evaluate on the **Civil** (`แพ่ง`) law type.
```python
import os
from dotenv import load_dotenv
from lawbuddy.eval import evaluate
from lawbuddy.rag import Hybrid

# Load pipeline
pipeline = Hybrid.from_openai_model(model="gpt-3.5-turbo")

# Load existing vector store
pipeline.load_vector_store(path="spaces/iterative_query_chunking")

# Get OpenAI API key
openai_key = os.getenv('OPENAI_API_KEY')

# Run evaluation
evaluate(pipeline, type_name='แพ่ง', model='gpt-3.5-turbo', openai_key=openai_key)
```



🔧 Advanced Configuration
------------------
The system supports various configurations for both OpenAI and local models. You can customize:

Chunk sizes for document processing
Vector store parameters
Model-specific settings
Query processing parameters

🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.
📝 License
MIT License
📬 Contact
For support or queries, please open an issue in the GitHub repository.

Made with ❤️ for the LawBuddy team.
