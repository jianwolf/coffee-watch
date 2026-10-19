# Shopify Storefront Bot-Gating (first observed 2026-06-10)

Status: **escalated 2026-10-18 — the catalog API is now gated too.** Re-check with each
run; update this doc when behavior changes.

## Escalation, 2026-10-18

The 2026-10-18 run saw 429 on `GET /collections/{handle}/products.json` itself — the
endpoint that was explicitly unaffected in June. 13 hosts returned 429 for the catalog
API; retries recovered two of them (Passenger, Black & White), and **11 of 17 roasters
finished the run with zero products**:

Prodigal, Onyx, Hydrangea, Flower Child, La Cabra, Regalia, Ilse, Dayglow, Heart,
George Howell, Olympia.

Run totals: 127×200, 37×429, 2×401, 104 products from 6 roasters, 125.6 s. Compare
2026-09-01: 371 products from 17/17 roasters.

### It is not the egress IP (tested 2026-10-19)

A residential scrape and a datacenter one, started 90 seconds apart against the
same 17 hosts, landed in the same place: **4 of 17** from the home IP and
**3 of 17** from an Anthropic cloud VM. Whatever is throttling this is not
scoring the client IP's reputation.

What does predict coverage is how recently the hosts were hit. Across one
evening, with six full scrapes between the two IPs:

| Time (UTC) | Where | Coverage |
| :--- | :--- | :--- |
| 2026-10-18 21:46 | residential, first scrape since 2026-09-01 | 6/17, 104 products |
| 2026-10-18 22:08 | residential | 8/17, 140 products |
| 2026-10-19 01:17 | cloud | 9/17, 173 products |
| 2026-10-19 02:34 | residential | 4/17, 59 products |
| 2026-10-19 02:35 | cloud | 3/17 |

The surviving set is close to random per run: of the two simultaneous runs, only
Hydrangea came back on both. That is the shape of a token bucket, not an
allow/deny list, and it means a single bad run says nothing about a store.

It also means our own testing is part of what we measured. Six scrapes in five
hours is far outside this project's stated low-frequency premise, and coverage
roughly halved over that window on both IPs. Treat a cold run after a quiet
night as the only honest baseline, and do not re-scrape to "check" a bad result.

Coverage is a per-run lottery, not a stable per-store flag. A second full run two hours
later got 140 products from 8/17 — a *different* survivor set (Onyx, Olympia, Hydrangea
and Ilse came back; Black & White, Passenger and Proud Mary dropped out). A third,
two-roaster run another twenty minutes on got 429 for both, including Little Wolf, which
had served 15 products in the first run. Caveat on that third data point: it was the
third scrape from one IP in one evening, against a project whose whole premise is
low-frequency monitoring, so it is not evidence of a cold-IP baseline. The first run is:
it was the first scrape since 2026-09-01 and 11 of 17 roasters already returned nothing.

Consequences, in order of importance:

1. **The "not affected" column below no longer holds.** Catalog coverage, not just
   storefront verification, is now degraded. A gated roaster contributes nothing.
2. **A degraded run looks like a quiet market.** The run still exited 0 and wrote a
   valid catalog. That is why `coffee_watch/runner.py` now exits `3` on a zero-product
   run and logs partial coverage as a warning, and why
   `python -m coffee_watch.brief` leads with a coverage header naming every empty
   roaster. Any report that omits coverage is wrong, not merely incomplete.
3. **Catalog-to-catalog deltas need a coverage guard.** A roaster that returned nothing
   makes its whole previous lineup look sold out. `brief.compute_delta` excludes
   products from roasters with no products this run and counts them as unknown.
4. The response policy below is unchanged: no UA spoofing, record the evidence, back
   off. The fast-skip gate still applies to page fetches only; catalog API and
   robots.txt requests deliberately do not consult it, so a transient API 429 is still
   retried rather than skipped for the rest of the run.

## What happened

Between the 2026-06-09 and 2026-06-10 scrape runs (~12 hours apart), 9 of the 14
Shopify-platform roasters began returning **HTTP 429 (no `Retry-After` header) for
product HTML pages** when fetched with the project's bot User-Agent
(`CoffeeWatch-Bot/1.0 ...`):

- Onyx Coffee Lab, Dayglow Coffee, Prodigal Coffee, Proud Mary Coffee,
  Hydrangea Coffee Roasters, Little Wolf Coffee, Regalia Coffee, Ilse Coffee,
  George Howell Coffee

The remaining Shopify stores (SEY, Black & White, Passenger, Olympia, Heart,
Flower Child) still served product pages normally in the same run — consistent with a
per-store feature flag / staged rollout.

**The catalog API was not affected.** `GET /collections/{handle}/products.json`
(the scraper's primary data source) worked on every gated store: the 2026-06-10 run
collected 369 products across 17/17 roasters with per-roaster counts matching the
previous run exactly.

## Evidence that this is UA classification, not our behavior

- A **single isolated request** (fresh connection, 3 s apart, one request per host) to
  three gated stores returned an immediate 429. A rate limit cannot be exceeded by one
  request; this is classification of the bot User-Agent (or fingerprint) on HTML routes.
- Stores never touched by ad-hoc testing (Onyx, George Howell, Proud Mary) were gated
  identically to ones that were.
- The scraper's profile is polite: 1 concurrent request per host, 0.7–2.0 s jitter
  between same-host requests, robots.txt checked before every fetch, `Retry-After`
  honored, honest UA with a contact URL. The same volume on 2026-06-09 triggered nothing.
- Caveat: the isolated-request tests ran minutes after a full scrape from the same IP,
  so short-lived IP-level bot scoring could not be fully ruled out on day one. A run on
  a later day with cold IP history distinguishes the two.

## What Shopify's own policy says

The gating **contradicts Shopify's published policy**, which explicitly permits
read-only access of exactly the kind this project performs:

- `robots.txt` (new platform boilerplate, first seen 2026-06-10) begins:
  *"Public product, collection, page, blog, policy, cart, and localized HTML is
  crawlable."* The machine-readable rules are `User-agent: * / Allow: /` with only the
  standard disallows (`/admin`, `/cart`, `/checkout`, `/orders`, `/account`,
  faceted-search crawl traps). Nothing prohibits product-page scraping.
- The robots.txt now links to `agents.md`, `.well-known/ucp` (UCP discovery), and a
  UCP/MCP endpoint, and includes *"Checkouts are for humans"* language — all of which
  concerns **transactions** (automated checkout/payment), not reading.
- `agents.md` contains a **"Read-Only Browsing (No Authentication Required)"** section
  that explicitly lists the sanctioned endpoints for non-transacting agents:
  - `GET /products/{handle}` (product page)
  - `GET /products/{handle}.json` (per-product JSON)
  - `GET /collections/{handle}/products.json` (collection catalog — our endpoint)
- The only scraping-adjacent requests in their docs: transacting agents "should prefer
  the Shop skill over screen-scraping," and all agents should back off on 429 (we do).

Most likely explanations (not mutually exclusive): a new edge-enforcement layer that
allowlists verified search engines and blocks unknown bot UAs on HTML routes (rolled
out before the docs and the firewall converged), and/or deliberate steering of agents
toward Shopify-controlled structured channels (products.json, UCP/MCP, shop.app) while
quietly throttling HTML.

## Impact on coffee-watch data

The table below describes the 2026-06-10 state. As of 2026-10-18 the right-hand column
is no longer accurate for gated stores: the catalog API is gated for most of them, so
they contribute no products at all. See the escalation section above.

| Affected | Not affected |
| :--- | :--- |
| Storefront verification pages (`verify_variant_pages`): no `visible_variant` price confirmation on gated stores (151 → 88 between runs) | Catalog API: products, variants, prices, availability (`price_source=variant` / `variant_api_grams` fallback works) |
| Stock counts / preorder ship dates scraped from page text | Product counts and catalog coverage (exact parity with previous run) |
| Page-text tasting notes for stores whose notes came from storefront pages (e.g. Onyx) | Wix stores (Memli) and non-gated Shopify stores |

429 is deliberately **not** in `STOREFRONT_UNAVAILABLE_STATUS_CODES` (401/403/404/410),
so gated products are *not* marked unavailable — they keep their API evidence and the
error is recorded per product in the catalog JSON.

## Response policy

1. **Do not spoof a browser User-Agent.** The codebase deliberately strips custom
   `User-Agent` headers (`merge_headers`); evading classification would violate the
   project's politeness stance.
2. **Re-check before adapting.** The gate is per-store and newly rolled out; status may
   change daily. Probe: a single `GET https://<store>/products/<handle>` with the bot
   UA; 429 = still gated.
3. If gating persists, preferred adaptations (in order):
   - **Fast-skip** — *implemented 2026-07-01*: after 3 consecutive final 429s from a
     host, the run skips that host's remaining page fetches (`Host429Gate` in
     `coffee_watch/http_limits.py`, shared across the run via `RunContext.page_gate`).
     Each skipped product records `storefront page skipped: host bot-gated by
     consecutive 429s`, and the roaster status note reports the skip counts. Catalog
     API and robots.txt requests never consult or feed the gate.
   - **Sanctioned JSON fallback**: fetch `GET /products/{handle}.json` for per-product
     data (note: it reports API variants, so it cannot fully replace *visible*-variant
     verification).
   - **UCP/MCP `search_catalog`**: the fully sanctioned agent channel; likely reflects
     buyable-only items, which could replace storefront verification properly.
4. Escalation contact published by Shopify: `bots@shopify.com`.

## Log signature

```
WARNING | coffee_watch | Non-200 response 429 for product page https://...
INFO | coffee_watch | Skipping fetch of https://...: host bot-gated by consecutive 429s.
```
plus per-product catalog errors: `storefront page returned status 429` for the pages
that were tried, and `storefront page skipped: host bot-gated by consecutive 429s`
for the fast-skipped remainder.
