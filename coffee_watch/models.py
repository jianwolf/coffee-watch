from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from pydantic import BaseModel, Field, confloat


@dataclass(frozen=True)
class RoasterSource:
    name: str
    base_url: str
    products_path: str = "/products.json"
    enabled: bool = True
    products_type: str = "auto"  # auto|json|html
    products_parser: Optional[str] = None
    jitter_multiplier: float = 1.0
    products_headers: dict[str, str] = field(default_factory=dict)
    products_params: dict[str, str] = field(default_factory=dict)
    product_page_headers: dict[str, str] = field(default_factory=dict)
    product_link_patterns: tuple[str, ...] = ("/products/", "/product/")
    product_link_exclude_patterns: tuple[str, ...] = ()
    product_url_template: Optional[str] = None
    json_items_path: Optional[tuple[str, ...]] = None
    product_fields: Optional["ProductFieldConfig"] = None
    pagination: Optional["PaginationConfig"] = None
    max_products: Optional[int] = None
    page_text_stop_phrases: tuple[str, ...] = ()
    include_tags: tuple[str, ...] = ()
    exclude_tags: tuple[str, ...] = ()
    include_product_types: tuple[str, ...] = ()
    exclude_product_types: tuple[str, ...] = ()
    exclude_title_keywords: tuple[str, ...] = ()

    @property
    def products_url(self) -> str:
        from urllib.parse import urljoin

        base = self.base_url if self.base_url.endswith("/") else f"{self.base_url}/"
        return urljoin(base, self.products_path.lstrip("/"))


@dataclass(frozen=True)
class ProductCandidate:
    product_id: str
    name: str
    url: str
    source: str
    list_price: str = ""
    list_badge: str = ""
    body_html: str = ""
    variants: tuple["VariantInfo", ...] = ()


@dataclass(frozen=True)
class VariantInfo:
    title: str
    price: str
    grams: int
    available: bool


class CoffeeScore(BaseModel):
    score: confloat(ge=0, le=10)
    worth_trying: bool
    summary: str = Field(..., description="Short, neutral summary of the coffee.")
    rationale: str = Field(..., description="Brief, critical explanation for the score.")
    rarity_signals: list[str] = Field(default_factory=list)
    reputation_signals: list[str] = Field(default_factory=list)
    confidence: confloat(ge=0, le=1) = 0.5


class CoffeeEvaluation(BaseModel):
    product_id: str = Field(..., description="Use the provided product_id exactly.")
    name: str
    url: str
    score: confloat(ge=0, le=10)
    worth_trying: bool
    summary: str
    rationale: str
    rarity_signals: list[str] = Field(default_factory=list)
    reputation_signals: list[str] = Field(default_factory=list)
    confidence: confloat(ge=0, le=1) = 0.5


class CoffeeBatchResult(BaseModel):
    evaluations: list[CoffeeEvaluation]


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
    page_size_param: Optional[str] = None
    page_size: Optional[int] = None
    stop_on_empty: bool = True
