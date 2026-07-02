# Shopify Storefront Bot-Gating (first observed 2026-06-10)

Status: **active as of 2026-06-10**. Re-check with each run; update this doc when behavior changes.

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
