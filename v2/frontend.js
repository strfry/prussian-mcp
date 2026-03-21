/**
 * Frontend für FastAPI Embeddings-Backend
 * Direkt gegen http://localhost:8000 oder deine Domain
 */

class DictionarySearch {
    constructor(apiUrl = "http://localhost:8000") {
        this.apiUrl = apiUrl;
        this.searchInput = document.getElementById("search-input");
        this.resultsContainer = document.getElementById("results");
        this.loadingSpinner = document.getElementById("loading");
        this.statsDiv = document.getElementById("stats");
        
        this.debounceTimer = null;
        this.debounceDelay = 300;
        
        this.init();
    }
    
    async init() {
        // Lade Stats
        this.loadStats();
        
        // Event Listener
        if (this.searchInput) {
            this.searchInput.addEventListener("input", (e) => {
                clearTimeout(this.debounceTimer);
                this.debounceTimer = setTimeout(() => {
                    this.search(e.target.value);
                }, this.debounceDelay);
            });
            
            this.searchInput.addEventListener("keydown", (e) => {
                if (e.key === "Enter") {
                    clearTimeout(this.debounceTimer);
                    this.search(e.target.value);
                }
            });
        }
    }
    
    async loadStats() {
        try {
            const response = await fetch(`${this.apiUrl}/stats`);
            const data = await response.json();
            
            if (this.statsDiv) {
                this.statsDiv.innerHTML = `
                    📖 <strong>${data.total_entries}</strong> Einträge | 
                    Model: ${data.model} | 
                    ${data.cached ? "✓ Gecacht" : "○ Live"}
                `;
            }
        } catch (err) {
            console.error("Stats load failed:", err);
        }
    }
    
    async search(query) {
        if (!query.trim()) {
            this.clear();
            return;
        }
        
        this.showLoading(true);
        
        try {
            const response = await fetch(`${this.apiUrl}/search`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    query: query,
                    top_k: 10
                })
            });
            
            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || "API error");
            }
            
            const data = await response.json();
            this.displayResults(data.results, query, data.count);
        } catch (err) {
            this.showError(`Error: ${err.message}`);
        } finally {
            this.showLoading(false);
        }
    }
    
    displayResults(results, query, count) {
        if (!this.resultsContainer) return;
        
        if (!results || results.length === 0) {
            this.resultsContainer.innerHTML = `
                <div class="no-results">
                    Keine Ergebnisse für: <strong>${this.escapeHtml(query)}</strong>
                </div>
            `;
            return;
        }
        
        let html = `<div class="results-header">${count} Ergebnis${count > 1 ? "se" : ""}</div>`;
        
        results.forEach((result, idx) => {
            html += this.renderResult(result, idx);
        });
        
        this.resultsContainer.innerHTML = html;
        this.attachClickHandlers();
    }
    
    renderResult(result, idx) {
        const p = result.prussian || "?";
        const e = result.english || "—";
        const d = result.german || "—";
        const l = result.lithuanian || "—";
        const r = result.russian || "—";
        const pl = result.polish || "—";
        const score = (result.score * 100).toFixed(1);
        
        return `
            <div class="entry" data-index="${idx}">
                <div class="entry-header" onclick="this.parentElement.querySelector('.entry-details').style.display = 
                    this.parentElement.querySelector('.entry-details').style.display === 'none' ? 'block' : 'none';">
                    <span class="entry-word">${this.escapeHtml(p)}</span>
                    <span class="entry-en">${this.escapeHtml(e)}</span>
                    <span class="entry-de">${this.escapeHtml(d)}</span>
                    <span class="entry-score">${score}%</span>
                    <span class="entry-toggle">▼</span>
                </div>
                <div class="entry-details" style="display: none;">
                    <table class="entry-table">
                        <tr><td>Prußisch:</td><td><strong>${this.escapeHtml(p)}</strong></td></tr>
                        <tr><td>English:</td><td>${this.escapeHtml(e)}</td></tr>
                        <tr><td>Deutsch:</td><td>${this.escapeHtml(d)}</td></tr>
                        <tr><td>Lietuvių:</td><td>${this.escapeHtml(l)}</td></tr>
                        <tr><td>Русский:</td><td>${this.escapeHtml(r)}</td></tr>
                        <tr><td>Polski:</td><td>${this.escapeHtml(pl)}</td></tr>
                        <tr><td>Score:</td><td><code>${result.score.toFixed(4)}</code></td></tr>
                    </table>
                </div>
            </div>
        `;
    }
    
    attachClickHandlers() {
        document.querySelectorAll(".entry-header").forEach(header => {
            header.addEventListener("click", (e) => {
                const details = header.nextElementSibling;
                const toggle = header.querySelector(".entry-toggle");
                
                if (details.style.display === "none") {
                    details.style.display = "block";
                    toggle.textContent = "▲";
                } else {
                    details.style.display = "none";
                    toggle.textContent = "▼";
                }
            });
        });
    }
    
    showLoading(show) {
        if (this.loadingSpinner) {
            this.loadingSpinner.style.display = show ? "inline-block" : "none";
        }
    }
    
    showError(message) {
        if (this.resultsContainer) {
            this.resultsContainer.innerHTML = `<div class="error">⚠️ ${this.escapeHtml(message)}</div>`;
        }
    }
    
    clear() {
        if (this.resultsContainer) {
            this.resultsContainer.innerHTML = "";
        }
    }
    
    escapeHtml(text) {
        if (!text) return "";
        const map = {
            "&": "&amp;",
            "<": "&lt;",
            ">": "&gt;",
            '"': "&quot;",
            "'": "&#039;"
        };
        return text.replace(/[&<>"']/g, m => map[m]);
    }
}

// Initialize on page load
document.addEventListener("DOMContentLoaded", () => {
    window.dictSearch = new DictionarySearch();
});
