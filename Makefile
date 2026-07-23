VENV_PYTHON := .venv/bin/python
DICT_FILE := data/twanksta_entries.json

# --- Chunk build inputs (siblings; see repos.tsv binder layout) ---
DICT_SRC    := ../corpus/parsed/twanksta_entries.json
LINKS       := ../fst/build/links.json
CHUNKS      := ../embeddings/data/embedding_chunks.jsonl
CHUNK_STORE := ../embeddings/data/embeddings_chunks_fastembed
CHUNK_MODEL := sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
BUILD_CHUNKS := .venv/bin/prussian-embeddings-build-chunks
GEN          := .venv/bin/prussian-embeddings-generate

.PHONY: all download embeddings chunks clean help

all: download

# --- Download ---

# Prefer the local corpus checkout (offline); fall back to the release tarball.
$(DICT_FILE):
	mkdir -p data
	@if [ -f $(DICT_SRC) ]; then \
		echo "cp $(DICT_SRC) -> $@ (offline)"; cp $(DICT_SRC) $@; \
	else \
		curl -L -o $@ https://github.com/strfry/prussian-corpus/releases/latest/download/$(notdir $@); \
	fi

download: $(DICT_FILE)

# --- Embeddings ---

embeddings: $(DICT_FILE)
	$(VENV_PYTHON) scripts/generate_embeddings.py

# --- Chunk store (single retrieval path since the chunk-only collapse) ---

# dictionary + FST links -> ref-clustered chunk JSONL
$(CHUNKS): $(DICT_FILE) $(LINKS)
	$(BUILD_CHUNKS) \
	  --dictionary $(DICT_FILE) --links $(LINKS) \
	  --out $(CHUNKS) --langs engl,miks,leit,latt

# JSONL -> fastembed chunk store (strategy=chunks; records carry text+members)
chunks: $(CHUNKS)
	$(GEN) \
	  --chunks $(CHUNKS) --backend sentence-transformers --model $(CHUNK_MODEL) \
	  --out $(CHUNK_STORE) --passage-prefix "Document: " --query-prefix "Query: "

# --- Cleanup ---

clean:
	rm -f $(DICT_FILE)
	rm -f embeddings/*.entries.json embeddings/*.embeddings.npy embeddings/*.meta.json

help:
	@echo 'Available targets:'
	@echo '  make download   - Get twanksta_entries.json (local corpus copy or release)'
	@echo '  make embeddings - Generate entry embeddings from dictionary'
	@echo '  make chunks     - Build the fastembed chunk store (dictionary + FST links)'
	@echo '  make all        - download'
	@echo '  make clean      - Remove downloaded data and generated files'
