# Test Plan

## Bug: Dictionary Tooltip Only Shows First Response

### Manual Test
1. Open browser dev console (F12)
2. Navigate to https://strfry.org/chatbot.html
3. **First message:**
   - Type: "Sveikas!"
   - Expected: Response shows "📖 N Wörterbucheinträge" tooltip
   - Actual: ✓ Works

4. **Second message:**
   - Type: "Kāi laimē?"
   - Expected: Response shows "📖 N Wörterbucheinträge" tooltip
   - Actual: ✗ No tooltip visible

5. **Check console for errors:**
   - Look for JavaScript errors in console
   - Check if `addMsg()` is being called
   - Verify `lookup_result.words` contains data

### CLI Test
```bash
cd /home/strfry/prussian-dictionary

# Test lookup function with multiple queries
node test_lookup.js "Sveikas"
node test_lookup.js "kail"
node test_lookup.js "antras"
```

### Code Review
- Check `send()` function - does it call `addMsg()` with words on every message?
- Check `addMsg()` function - is it correctly appending tooltip div?
- Check CSS - is `.dict-hits` tooltip being hidden/overwritten?
- Check history - does it contain all messages?

### Expected Behavior
- Every bot message should show tooltip with dictionary entry names
- Tooltip should be clickable/hoverable
- Works consistently across multiple exchanges

### Files to Check
- `/home/strfry/prussian-dictionary/chatbot.html` (main file)
  - `send()` function ~line 450
  - `addMsg()` function ~line 380
  - `.dict-hits` CSS styling ~line 60
