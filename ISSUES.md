# Bug Tracker

## Known Issues

### 🔴 CRITICAL
- [ ] **Dictionary tooltip only shows for first response**
  - Status: Reported
  - Description: Tooltip with dictionary hits (📖 N Wörterbucheinträge) only appears on first chatbot answer. Subsequent answers don't show tooltip.
  - Steps to reproduce:
    1. Open https://strfry.org/chatbot.html
    2. Type message → First response shows tooltip with dictionary hits ✓
    3. Type another message → Second response NO tooltip ✗
  - Expected: All bot responses should show tooltip with dictionary hits
  - Root cause: Unknown - likely issue in `addMsg()` function or tooltip rendering logic
  - Files affected: `chatbot.html` (addMsg function)
  - Priority: HIGH (affects UI feedback)
  - Test needed: See TEST_PLAN.md

## In Progress
- [ ] Language-based prefix matching (short words < 4 chars)
  - Implemented, deployed, needs testing

## Completed
- [x] Language selector (Deutsch/Lietuvių) with auto-detection
- [x] Relevance-based search ranking
- [x] 64 common short words in system prompt
- [x] Prefix matching for short queries (< 4 chars)
