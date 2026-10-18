# Das MCP-Repo hält keine eigenen Datendateien mehr:
#
# - Dictionary:  ../corpus/parsed/twanksta_dmlex.nvh (lean NVH, kanonische
#   Quelle; geschrieben vom Kompressor im fst-Repo)
# - Embedding-Stores: ../embeddings/data/ (Pipeline: ../embeddings/README.md)
#
# Nach Rebuild der Stores/Des Dictionaries: MCP-Server neu starten —
# die Engine lädt alles nur beim Start.

# --- Chunk build inputs (siblings; see repos.tsv binder layout) ---
DICT_SRC    := ../corpus/parsed/twanksta_dmlex.nvh
LINKS       := ../fst/build/links.json
CHUNKS      := ../embeddings/data/embedding_chunks.jsonl
CHUNK_STORE := ../embeddings/data/embeddings_chunks_fastembed
CHUNK_MODEL := sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
BUILD_CHUNKS := .venv/bin/prussian-embeddings-build-chunks
GEN          := .venv/bin/prussian-embeddings-generate

.PHONY: chunks help

# --- Chunk store (single retrieval path since the chunk-only collapse) ---

# dictionary + FST links -> ref-clustered chunk JSONL
$(CHUNKS): $(DICT_SRC) $(LINKS)
	$(BUILD_CHUNKS) \
	  --dictionary $(DICT_SRC) --links $(LINKS) \
	  --out $(CHUNKS) --langs en,de,lt,lv

# JSONL -> fastembed chunk store (strategy=chunks; records carry text+members)
chunks: $(CHUNKS)
	$(GEN) \
	  --chunks $(CHUNKS) --backend sentence-transformers --model $(CHUNK_MODEL) \
	  --out $(CHUNK_STORE) --passage-prefix "Document: " --query-prefix "Query: "

help:
	@echo 'Keine Daten-Kopien im Repo.'
	@echo '  make chunks         - Chunk-Store bauen (Dictionary + FST-Links)'
	@echo '  Dictionary-Neubau:  lean NVH via ../corpus (twanksta-dmlex) + ../fst (compress_forms)'
	@echo '  Store-Pipeline:     siehe ../embeddings/README.md ("Ref-clustered chunks")'
