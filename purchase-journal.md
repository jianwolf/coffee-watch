# Purchase Journal / 购买日志

User-curated state for the coffee-scout skill. The skill reads this file at the start of
every scouting session and uses it three ways:

1. **Standing preferences** calibrate recommendation ranking (this replaces guessing; the
   skill must not assume preferences that are not written here or stated in the session).
2. **Watchlist** items are checked against the fresh catalog first: report restocks,
   price changes, or sell-outs before anything else.
3. **Purchase log** entries (with taste verdicts) personalize future reports: lean toward
   what worked, away from what did not, and avoid re-recommending recent purchases.

The `Last session` date also anchors the "new since last purchase" window: the skill
derives newness from each product's `first_seen_at` relative to this date, not just the
scraper's default 7-day flag.

After a purchase is decided in a session, the skill should append a Purchase log entry
(items, size-aware prices, URLs) and leave `Verdict` blank for the user to fill after
tasting. The skill may update this file; it should never delete user-written content.

---

## Standing preferences / 固定偏好

- Buying cadence: ~2 bags from one roaster, about once per month.
- What I actually want to hear about: coffee that would be a pain to miss -
  very rare, or a genuinely great flavor. Not rotation, not restocks, not
  another washed Ethiopia. If nothing qualifies on a given day, say so in one
  line; I would rather read "nothing today" than a menu.
- Tone: talk to me like a friend texting, not like a report.
- How recently something was listed does not matter to me. If it tastes good I
  do not care whether it arrived this week or two months ago. Only use the
  listing date to avoid telling me about the same coffee twice.
- Do not use price as a reason to recommend or skip something. I am not rich
  and I do not look down on cheap coffee - a bargain is not a reason to try a
  coffee, and an ordinary price is not a reason to leave one out. Show me the
  price and let me decide.
- What I want ranked on: how it tastes, and how genuinely rare it is. Those two
  only.
- (add taste, budget or roast preferences here as they come up)

## Watchlist / 关注清单

<!-- One per line: product or roaster + what to watch for. Example:
- George Howell Gesha Village Lot 47 — tell me when restocked
-->

(empty)

## Purchase log / 购买记录

Last session: (none yet)

<!-- Entry template:
### 2026-06-09 — Roaster Name
- Coffee Name — 250 g = $34.25 — https://example.com/products/x
- Coffee Name — 8oz = $24.00 — https://example.com/products/y
- Verdict: (fill after tasting: loved / fine / not again, and why)
-->
