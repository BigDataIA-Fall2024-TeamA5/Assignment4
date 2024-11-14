from langraph import Graph, Node
from document_agent import DocumentAgent
from arxiv_agent import ArxivAgent
from web_search_agent import WebSearchAgent
from rag_agent import RAGAgent

def create_research_graph():
    graph = Graph()

    # Create nodes for each agent
    document_node = Node(DocumentAgent().run, name="document_selector")
    arxiv_node = Node(ArxivAgent().run, name="arxiv_search")
    web_search_node = Node(WebSearchAgent().run, name="web_search")
    rag_node = Node(RAGAgent().run, name="rag_generator")

    # Add nodes to the graph
    graph.add_node(document_node)
    graph.add_node(arxiv_node)
    graph.add_node(web_search_node)
    graph.add_node(rag_node)

    # Define edges (connections between agents)
    graph.add_edge(document_node, arxiv_node)
    graph.add_edge(document_node, web_search_node)
    graph.add_edge(arxiv_node, rag_node)
    graph.add_edge(web_search_node, rag_node)

    # Define the entry point
    graph.set_entry_point(document_node)

    return graph

def run_research_graph(document, question):
    graph = create_research_graph()
    results = graph.run({"document": document, "question": question})
    return results