# Reranker Integration für prussian-mcp

## API-Endpoint
```
URL: localhost:8001/v3/rerank
Model: "BAAI/bge-reranker-large"
```

## Request-Format
```json
{
  "model": "BAAI/bge-reranker-large",
  "query": "<suchanfrage>",
  "documents": ["doc1", "doc2", ...]
}
```

## Response-Format
```json
{
  "results": [
    {"index": 0, "relevance_score": 0.99},
    {"index": 1, "relevance_score": 0.62},
    ...
  ]
}
```

## Implementierung
1. Immer 2-3x mehr Ergebnisse initialholen als angefragt
2. Query + Dokumente an Reranker senden
3. Nach relevance_score absteigend sortieren
4. Auf angefragte Anzahl kürzen

## Python Beispiel
```python
import httpx

async def rerank_results(query: str, documents: list[str], top_k: int = None) -> list[dict]:
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8001/v3/rerank",
            json={
                "model": "BAAI/bge-reranker-large",
                "query": query,
                "documents": documents
            }
        )
        results = response.json()["results"]
    
    sorted_results = sorted(results, key=lambda x: x["relevance_score"], reverse=True)
    
    if top_k:
        sorted_results = sorted_results[:top_k]
    
    return sorted_results
```

## Testing
- Erst Validierung dass bestehende Suche nicht schlechter wird
- Dann schrittweise Aktivierung mit Fallback