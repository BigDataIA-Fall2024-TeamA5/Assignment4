import arxiv
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import spacy
from nltk.corpus import wordnet as wn

# Load NLP models
nlp = spacy.load("en_core_web_sm")

# Load the pre-trained language model for embedding generation
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

# Function to convert text to embeddings
def get_text_embeddings(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state.mean(dim=1)
    return embeddings.squeeze().numpy()

# Synonym Expansion Function
def expand_query_with_synonyms(query):
    doc = nlp(query)
    expanded_terms = set()

    # Extract nouns and important words from the query
    for token in doc:
        if token.pos_ in ["NOUN", "PROPN"] and token.text not in expanded_terms:
            expanded_terms.add(token.text)

            # Find synonyms using WordNet
            for syn in wn.synsets(token.text):
                for lemma in syn.lemmas():
                    expanded_terms.add(lemma.name().replace("_", " "))

    # Formulate expanded query
    expanded_query = query + " " + " ".join(expanded_terms)
    return expanded_query

# Perform the search on arXiv
def arxiv_agent_search(query, num_results=10):
    # Expand the query with synonyms
    expanded_query = expand_query_with_synonyms(query)
    print(f"Expanded Query: {expanded_query}")

    # Search on arXiv using the expanded query
    search = arxiv.Search(
        query=expanded_query,
        max_results=num_results,
        sort_by=arxiv.SortCriterion.Relevance
    )
    
    # Collect metadata for each result
    papers = []
    for result in search.results():
        paper_info = {
            "title": result.title,
            "summary": result.summary,
            "pdf_url": result.pdf_url
        }
        papers.append(paper_info)
    
    return papers

# Re-rank papers based on cosine similarity between query and summary embeddings
def rerank_papers_by_relevance(query, papers):
    # Generate embedding for the query
    query_embedding = get_text_embeddings(query)
    
    # Calculate similarity scores and rank papers
    ranked_papers = []
    for paper in papers:
        # Generate embedding for each paper's summary
        summary_embedding = get_text_embeddings(paper['summary'])
        
        # Calculate cosine similarity between query and summary embeddings
        similarity_score = cosine_similarity([query_embedding], [summary_embedding])[0][0]
        
        # Add paper and its similarity score to the list
        ranked_papers.append((paper, similarity_score))
    
    # Sort papers by similarity score in descending order
    ranked_papers.sort(key=lambda x: x[1], reverse=True)
    
    # Return only the papers, sorted by relevance
    return [paper for paper, score in ranked_papers]

# Chatbot Functionality
def chatbot_search_and_recommend(user_question):
    # Step 1: Retrieve papers from arXiv
    print("Searching for relevant research papers on arXiv...")
    papers = arxiv_agent_search(query=user_question, num_results=10)

    # Step 2: Re-rank papers based on relevance to the query
    ranked_papers = rerank_papers_by_relevance(user_question, papers)

    # Prepare the response
    response = "\nHere are the most relevant research papers for your query:\n"
    for i, paper in enumerate(ranked_papers, 1):
        response += f"\nRank {i}:\n"
        response += f"Title: {paper['title']}\n"
        response += f"Summary: {paper['summary']}\n"
        response += f"PDF URL: {paper['pdf_url']}\n"

    return response

# Example usage for chatbot
if __name__ == "__main__":
    # Simulate chatbot interaction
    user_question = input("Ask your question: ")
    response = chatbot_search_and_recommend(user_question)
    print(response)
