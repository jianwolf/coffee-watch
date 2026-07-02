from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class RoasterSource:
    name: str
    base_url: str
    platform: str = "unknown"
    products_path: str = "/products.json"
    enabled: bool = True
    products_type: str = "auto"  # auto|json|html
    products_parser: str | None = None
    jitter_multiplier: float = 1.0
    products_headers: dict[str, str] = field(default_factory=dict)
    products_params: dict[str, str] = field(default_factory=dict)
    product_page_headers: dict[str, str] = field(default_factory=dict)
    product_link_patterns: tuple[str, ...] = ("/products/", "/product/")
    product_link_exclude_patterns: tuple[str, ...] = ()
    product_url_template: str | None = None
    json_items_path: tuple[str, ...] | None = None
    product_fields: ProductFieldConfig | None = None
    pagination: PaginationConfig | None = None
    max_products: int | None = None
    page_text_stop_phrases: tuple[str, ...] = ()
    verify_variant_pages: bool = False
    include_tags: tuple[str, ...] = ()
    exclude_tags: tuple[str, ...] = ()
    include_product_types: tuple[str, ...] = ()
    exclude_product_types: tuple[str, ...] = ()
    exclude_title_keywords: tuple[str, ...] = ()


@dataclass(frozen=True)
class ProductCandidate:
    product_id: str
    name: str
    url: str
    source: str
    list_price: str = ""
    list_badge: str = ""
    body_html: str = ""
    variants: tuple[VariantInfo, ...] = ()
    visible_variant_titles: tuple[str, ...] = ()
    storefront_status: str = ""
    shopify_updated_at: str = ""
    shopify_published_at: str = ""


@dataclass(frozen=True)
class VariantInfo:
    title: str
    price: str
    grams: int
    available: bool


@dataclass(frozen=True)
class ProductFieldConfig:
    name_fields: tuple[str, ...] = ("title", "name")
    url_fields: tuple[str, ...] = ("url", "product_url")
    handle_fields: tuple[str, ...] = ("handle", "slug")
    id_fields: tuple[str, ...] = ("id", "product_id")


@dataclass(frozen=True)
class PaginationConfig:
    page_param: str = "page"
    start: int = 1
    max_pages: int = 1
    page_size_param: str | None = None
    page_size: int | None = None
    stop_on_empty: bool = True


@dataclass(frozen=True)
class RoasterRunStatus:
    """Structured record of a single roaster's scrape outcome."""

    roaster: str
    run_id: str
    status: str  # success | failure | empty | skipped
    attempts: int
    catalog_path: str | None
    products_found: int
    new_products: int
    fetched_pages: int
    platform: str
    classified_by_source: dict[str, int] = field(default_factory=dict)
    undated: int = 0
    outside_window: int = 0
    completed_at: str = ""
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "roaster": self.roaster,
            "run_id": self.run_id,
            "status": self.status,
            "attempts": self.attempts,
            "catalog_path": self.catalog_path,
            "products_found": self.products_found,
            "new_products": self.new_products,
            "fetched_pages": self.fetched_pages,
            "platform": self.platform,
            "classified_by_source": dict(self.classified_by_source),
            "undated": self.undated,
            "outside_window": self.outside_window,
            "completed_at": self.completed_at,
            "note": self.note,
        }
