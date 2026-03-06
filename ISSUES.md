# Bug Tracker

## Known Issues

None currently! ✅

## Recently Fixed
- [x] **Dictionary tooltip only shows for first response** - FIXED in commit 307d5e3
  - Root cause: Consolidating multiple JS files into single `chatbot.js` resolved the issue
  - The problem occurred when scripts were split across multiple files (chatbot-utils.js, chatbot-i18n.js, etc.)
  - Solution: Merged all logic into `/scripts/chatbot.js`
  - Verification: All tooltips now show correctly on all bot responses
  - Tested: ✅ Console logs visible, ✅ Tooltips display correctly for 1st, 2nd, 3rd+ messages

## Completed Features
- [x] Language selector (Deutsch/Lietuvių) with auto-detection
- [x] Relevance-based search ranking
- [x] 64 common short words in system prompt
- [x] Prefix matching for short queries (< 4 chars)
- [x] Dictionary tooltips on all bot responses
