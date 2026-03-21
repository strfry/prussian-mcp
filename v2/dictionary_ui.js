/**
 * Frontend-Integration für Embeddings-basierte Wörterbuch-Suche
 * Ruft PHP-API auf und zeigt Ergebnisse
 */

class DictionaryEmbeddingsUI {
    constructor(apiUrl = '/api_embeddings.php') {
        this.apiUrl = apiUrl;
        this.searchInput = document.getElementById('dict-search-input');
        this.resultsContainer = document.getElementById('dict-results');
        this.loadingSpinner = document.getElementById('dict-loading');
        
        this.debounceTimer = null;
        this.debounceDelay = 300; // ms
        
        this.setupEventListeners();
        this.loadStats();
    }
    
    setupEventListeners() {
        if (this.searchInput) {
            // Debounced input
            this.searchInput.addEventListener('input', (e) => {
                clearTimeout(this.debounceTimer);
                this.debounceTimer = setTimeout(() => {
                    this.search(e.target.value);
                }, this.debounceDelay);
            });
            
            // Enter-Taste: sofort suchen
            this.searchInput.addEventListener('keydown', (e) => {
                if (e.key === 'Enter') {
                    clearTimeout(this.debounceTimer);
                    this.search(e.target.value);
                }
            });
        }
    }
    
    async loadStats() {
        try {
            const response = await fetch(this.apiUrl, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ action: 'get_stats' })
            });
            const data = await response.json();
            
            if (data.success) {
                console.log(`📖 Dictionary: ${data.dictionary_entries} entries (${data.backend})`);
            }
        } catch (err) {
            console.error('Failed to load stats:', err);
        }
    }
    
    async search(query) {
        if (!query.trim()) {
            this.clear();
            return;
        }
        
        this.showLoading(true);
        
        try {
            const response = await fetch(this.apiUrl, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    action: 'search',
                    query: query,
                    top_k: 10
                })
            });
            
            const data = await response.json();
            
            if (data.success) {
                this.displayResults(data.results, query);
            } else {
                this.showError(data.error);
            }
        } catch (err) {
            this.showError(`Network error: ${err.message}`);
        } finally {
            this.showLoading(false);
        }
    }
    
    displayResults(results, query) {
        if (!this.resultsContainer) return;
        
        if (!results || results.length === 0) {
            this.resultsContainer.innerHTML = `
                <div class="dict-no-results">
                    Keine Einträge gefunden für: <strong>${this.escapeHtml(query)}</strong>
                </div>
            `;
            return;
        }
        
        let html = `
            <div class="dict-results-header">
                ${results.length} Ergebnis${results.length > 1 ? 'se' : ''} für: 
                <strong>${this.escapeHtml(query)}</strong>
            </div>
        `;
        
        results.forEach((item, index) => {
            const entry = item.entry;
            const score = item.score ? item.score.toFixed(3) : 'N/A';
            
            html += this.renderEntry(entry, score, index);
        });
        
        this.resultsContainer.innerHTML = html;
        this.attachEntryClickHandlers();
    }
    
    renderEntry(entry, score, index) {
        const prussian = entry.prussian || '?';
        const english = entry.english || '?';
        const german = entry.german || '?';
        const lithuanian = entry.lithuanian || '?';
        const russian = entry.russian || '?';
        const polish = entry.polish || '?';
        const id = entry.id || `entry_${index}`;
        
        return `
            <div class="dict-entry" data-id="${this.escapeHtml(id)}">
                <div class="dict-entry-header">
                    <div class="dict-entry-main">
                        <span class="dict-prussian">${this.escapeHtml(prussian)}</span>
                        <span class="dict-score">[${score}]</span>
                    </div>
                    <div class="dict-entry-toggle">▼</div>
                </div>
                
                <div class="dict-entry-details" style="display: none;">
                    <table class="dict-entry-table">
                        <tr>
                            <td class="dict-lang">English:</td>
                            <td class="dict-value">${this.escapeHtml(english)}</td>
                        </tr>
                        <tr>
                            <td class="dict-lang">Deutsch:</td>
                            <td class="dict-value">${this.escapeHtml(german)}</td>
                        </tr>
                        <tr>
                            <td class="dict-lang">Lietuvių:</td>
                            <td class="dict-value">${this.escapeHtml(lithuanian)}</td>
                        </tr>
                        <tr>
                            <td class="dict-lang">Русский:</td>
                            <td class="dict-value">${this.escapeHtml(russian)}</td>
                        </tr>
                        <tr>
                            <td class="dict-lang">Polski:</td>
                            <td class="dict-value">${this.escapeHtml(polish)}</td>
                        </tr>
                    </table>
                </div>
            </div>
        `;
    }
    
    attachEntryClickHandlers() {
        document.querySelectorAll('.dict-entry').forEach(entry => {
            entry.addEventListener('click', () => {
                const details = entry.querySelector('.dict-entry-details');
                const toggle = entry.querySelector('.dict-entry-toggle');
                
                const isOpen = details.style.display !== 'none';
                details.style.display = isOpen ? 'none' : 'block';
                toggle.textContent = isOpen ? '▼' : '▲';
            });
        });
    }
    
    showLoading(show) {
        if (this.loadingSpinner) {
            this.loadingSpinner.style.display = show ? 'inline-block' : 'none';
        }
    }
    
    showError(message) {
        if (this.resultsContainer) {
            this.resultsContainer.innerHTML = `
                <div class="dict-error">⚠️ ${this.escapeHtml(message)}</div>
            `;
        }
    }
    
    clear() {
        if (this.resultsContainer) {
            this.resultsContainer.innerHTML = '';
        }
    }
    
    escapeHtml(text) {
        const map = {
            '&': '&amp;',
            '<': '&lt;',
            '>': '&gt;',
            '"': '&quot;',
            "'": '&#039;'
        };
        return text.replace(/[&<>"']/g, m => map[m]);
    }
}

// Initialisiere beim Laden
document.addEventListener('DOMContentLoaded', () => {
    window.dictionaryUI = new DictionaryEmbeddingsUI();
});

// Export für Nutzung in anderen Skripten
if (typeof module !== 'undefined' && module.exports) {
    module.exports = DictionaryEmbeddingsUI;
}
