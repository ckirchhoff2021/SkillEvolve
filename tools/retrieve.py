import os
import faiss
from models import Embedding
from dotenv import load_dotenv

load_dotenv()


def retrieve(documents: list[str], query_document: str, top_k: int = 2):
    api_base = os.getenv("EMBED_BASE_URL")
    model_name = os.getenv("EMBED_MODEL")
    
    embed = Embedding(api_base, "Empty", model_name)
    doc_embeddings = embed.infer(documents)
    
    # faiss.normalize_L2(doc_embeddings)
    # print(doc_embeddings.shape, doc_embeddings.dtype)
    
    dimension = doc_embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(doc_embeddings)
    query_emb = embed.infer([query_document])
    
    # faiss.normalize_L2(query_emb)
    
    distances, indices = index.search(query_emb, top_k)
    outputs = list()
    for i in range(top_k):
        doc_id = indices[0][i]
        score = distances[0][i]
        outputs.append({
            'score': score,
            'document': documents[doc_id],
            'doc_idx': doc_id,
        })
    return outputs


