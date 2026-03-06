
<?php
header("Content-Type: application/json");
header("Access-Control-Allow-Origin: *");

define("HF_TOKEN", "HF_TOKEN_REDACTED");
define("MODEL",    "Qwen/Qwen2.5-72B-Instruct");  // z.B. "deepseek-ai/DeepSeek-V3" zum Wechseln

$url = "https://router.huggingface.co/v1/chat/completions";

// ── Eingehenden Body lesen und von Anthropic- auf OpenAI-Format umwandeln ──
$in = json_decode(file_get_contents("php://input"), true);
if (!$in) { http_response_code(400); echo json_encode(["error" => "No input"]); exit; }

// Anthropic schickt system + messages getrennt; OpenAI erwartet alles in messages
$messages = [];
if (!empty($in["system"])) {
    $messages[] = ["role" => "system", "content" => $in["system"]];
}
foreach (($in["messages"] ?? []) as $m) {
    $messages[] = ["role" => $m["role"], "content" => $m["content"]];
}

$body = json_encode([
    "model"       => MODEL,
    "messages"    => $messages,
    "max_tokens"  => $in["max_tokens"] ?? 1000,
    "temperature" => 0.7,
    "stream"      => false,
]);

// ── Request an HuggingFace ─────────────────────────────────────────────────
$ch = curl_init($url);
curl_setopt_array($ch, [
    CURLOPT_RETURNTRANSFER => true,
    CURLOPT_POST           => true,
    CURLOPT_POSTFIELDS     => $body,
    CURLOPT_HTTPHEADER     => [
        "Content-Type: application/json",
        "Authorization: Bearer " . HF_TOKEN,
    ],
]);

$resp = curl_exec($ch);
$code = curl_getinfo($ch, CURLINFO_HTTP_CODE);
curl_close($ch);

// ── Antwort von OpenAI- auf Anthropic-Format umwandeln (für chatbot.html) ──
$out = json_decode($resp, true);
if (isset($out["choices"][0]["message"]["content"])) {
    $text = $out["choices"][0]["message"]["content"];
    http_response_code(200);
    echo json_encode([
        "content" => [["type" => "text", "text" => $text]]
    ]);
} else {
    // Fehler durchreichen
    http_response_code($code);
    echo $resp;
}
