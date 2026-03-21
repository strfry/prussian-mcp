<?php
/**
 * API für Embeddings-basierte Wörterbuch-Suche
 * Ruft Python-Backend auf oder nutzt vorberechnete Embeddings
 * 
 * Verwendung:
 * POST /api_proxy.php
 * {
 *   "action": "search",
 *   "query": "son",
 *   "top_k": 10
 * }
 */

header('Content-Type: application/json; charset=utf-8');
header('Access-Control-Allow-Origin: *');

class EmbeddingsAPI {
    private $entries_cache = null;
    private $embeddings_cache = null;
    private $python_script = "/path/to/prussian_embeddings.py";
    private $embeddings_dir = "/path/to/embeddings";
    
    public function __construct() {
        // Versuche, vorberechnete Embeddings zu laden (schneller!)
        $this->load_cached_embeddings();
    }
    
    /**
     * Lade gespeicherte Embeddings (numpy .npy Format)
     */
    private function load_cached_embeddings() {
        $entries_file = $this->embeddings_dir . "/entries.json";
        
        if (file_exists($entries_file)) {
            $json_content = file_get_contents($entries_file);
            $this->entries_cache = json_decode($json_content, true);
        }
    }
    
    /**
     * Haupteinstiegspunkt für Anfragen
     */
    public function handle_request() {
        $input = json_decode(file_get_contents('php://input'), true);
        
        if (!$input || !isset($input['action'])) {
            return $this->error("Missing 'action' parameter");
        }
        
        $action = $input['action'];
        
        switch ($action) {
            case 'search':
                return $this->search($input['query'] ?? '', $input['top_k'] ?? 10);
            case 'search_by_embedding':
                return $this->search_by_embedding($input['embedding'] ?? [], $input['top_k'] ?? 10);
            case 'get_entry':
                return $this->get_entry($input['id'] ?? '');
            case 'get_stats':
                return $this->get_stats();
            default:
                return $this->error("Unknown action: $action");
        }
    }
    
    /**
     * Suche nach Text-Query
     * Konvertiert Query zu Embedding und findet ähnlichste Einträge
     */
    private function search(string $query, int $top_k = 10) {
        if (empty($query)) {
            return $this->error("Empty query");
        }
        
        // OPTION 1: Nutze Python-Subprocess für echte Embeddings
        if ($this->use_python_backend()) {
            return $this->search_via_python($query, $top_k);
        }
        
        // OPTION 2: Fallback - einfache Token-basierte Suche
        return $this->search_simple($query, $top_k);
    }
    
    /**
     * Rufe Python-Embeddings-Backend auf
     */
    private function search_via_python(string $query, int $top_k = 10) {
        $python_code = <<<'PYTHON'
import json
import sys
sys.path.insert(0, '/path/to/script')
from prussian_embeddings import PrussianDictionaryEmbeddings

embedder = PrussianDictionaryEmbeddings()
embedder.load_embeddings("/path/to/embeddings")

query = sys.argv[1]
top_k = int(sys.argv[2])
results = embedder.search(query, top_k)

output = [
    {
        "entry": entry,
        "score": float(score)
    }
    for entry, score in results
]

print(json.dumps(output, ensure_ascii=False))
PYTHON;
        
        $temp_file = tempnam(sys_get_temp_dir(), 'emb_');
        file_put_contents($temp_file, $python_code);
        
        $output = shell_exec(
            sprintf(
                "python3 %s %s %d 2>&1",
                escapeshellarg($temp_file),
                escapeshellarg($query),
                intval($top_k)
            )
        );
        
        unlink($temp_file);
        
        if ($output === null) {
            return $this->error("Python backend failed");
        }
        
        $results = json_decode($output, true);
        
        return [
            "success" => true,
            "query" => $query,
            "results" => $results
        ];
    }
    
    /**
     * Fallback: einfache Token-basierte Suche
     */
    private function search_simple(string $query, int $top_k = 10) {
        if (!$this->entries_cache) {
            return $this->error("No dictionary entries loaded");
        }
        
        $query_tokens = array_flip(preg_split('/\s+/', strtolower($query)));
        $results = [];
        
        foreach ($this->entries_cache as $entry) {
            $score = 0;
            
            // Zähle Token-Matches in allen Sprachen
            foreach (['prussian', 'english', 'german', 'lithuanian', 'russian', 'polish'] as $lang) {
                if (!isset($entry[$lang])) continue;
                
                $text_tokens = preg_split('/\s+/', strtolower($entry[$lang]));
                
                foreach ($text_tokens as $token) {
                    if (isset($query_tokens[$token])) {
                        $score += 1.0;
                    }
                    // Substring-Match (weniger Gewicht)
                    else if (strpos($token, $query) !== false || strpos($query, $token) !== false) {
                        $score += 0.3;
                    }
                }
            }
            
            if ($score > 0) {
                $results[] = [
                    "entry" => $entry,
                    "score" => $score
                ];
            }
        }
        
        // Sortiere nach Score (absteigend)
        usort($results, fn($a, $b) => $b['score'] <=> $a['score']);
        
        return [
            "success" => true,
            "query" => $query,
            "results" => array_slice($results, 0, $top_k),
            "method" => "simple_token_match"
        ];
    }
    
    /**
     * Suche nach bereits berechneten Embedding-Vektor
     */
    private function search_by_embedding(array $embedding, int $top_k = 10) {
        // Noch nicht implementiert - würde Vektoren-Ähnlichkeit rechnen
        return $this->error("search_by_embedding not yet implemented");
    }
    
    /**
     * Hole einen bestimmten Eintrag
     */
    private function get_entry(string $id) {
        if (!$this->entries_cache) {
            return $this->error("No dictionary entries loaded");
        }
        
        foreach ($this->entries_cache as $entry) {
            if (($entry['id'] ?? null) === $id) {
                return [
                    "success" => true,
                    "entry" => $entry
                ];
            }
        }
        
        return $this->error("Entry not found: $id");
    }
    
    /**
     * Gib Statistiken zurück
     */
    private function get_stats() {
        $count = $this->entries_cache ? count($this->entries_cache) : 0;
        
        return [
            "success" => true,
            "dictionary_entries" => $count,
            "backend" => $this->use_python_backend() ? "python_embeddings" : "simple_token",
            "cached" => $this->entries_cache !== null
        ];
    }
    
    private function use_python_backend(): bool {
        return function_exists('shell_exec') && 
               !in_array('shell_exec', explode(',', ini_get('disable_functions')));
    }
    
    private function error(string $message) {
        return [
            "success" => false,
            "error" => $message
        ];
    }
}

// Haupteinstiegspunkt
$api = new EmbeddingsAPI();
echo json_encode($api->handle_request(), JSON_UNESCAPED_UNICODE | JSON_PRETTY_PRINT);
