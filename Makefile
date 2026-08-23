# Das MCP-Repo hält keine eigenen Datendateien mehr:
#
# - Dictionary:  ../corpus/parsed/twanksta_entries.json (kanonische Quelle,
#   gebaut von scripts/twanksta_parse.py im corpus-Repo)
# - Embedding-Stores: ../embeddings/data/ (Pipeline: ../embeddings/README.md)
#
# Nach Rebuild der Stores/Des Dictionaries: MCP-Server neu starten —
# die Engine lädt alles nur beim Start.

.PHONY: help

help:
	@echo 'Keine lokalen Daten-Targets.'
	@echo '  Dictionary-Neubau:  make -C ../corpus parse (scripts/twanksta_parse.py)'
	@echo '  Store-Pipeline:     siehe ../embeddings/README.md ("Ref-clustered chunks")'
