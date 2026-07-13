VENV_PYTHON := .venv/bin/python
DICT_FILE := data/twanksta_entries.json

.PHONY: all download embeddings clean help

all: download

# --- Download ---

$(DICT_FILE):
	mkdir -p data
	curl -L -o $@ https://github.com/strfry/prussian-corpus/releases/latest/download/$(notdir $@)

download: $(DICT_FILE)

# --- Embeddings ---

embeddings: $(DICT_FILE)
	$(VENV_PYTHON) scripts/generate_embeddings.py

# --- Cleanup ---

clean:
	rm -f $(DICT_FILE)
	rm -f embeddings/*.entries.json embeddings/*.embeddings.npy embeddings/*.meta.json

help:
	@echo 'Available targets:'
	@echo '  make download   - Download twanksta_entries.json from latest release'
	@echo '  make embeddings - Generate embeddings from dictionary'
	@echo '  make all        - download + embeddings'
	@echo '  make clean      - Remove downloaded data and generated files'
