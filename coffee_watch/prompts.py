from __future__ import annotations

from typing import Any

from .models import ProductCandidate
from .text_utils import format_variant_lines, sanitize_html_to_text, sanitize_prompt_field


def normalize_language(value: str) -> str:
    normalized = value.strip().lower()
    if normalized in {"zh", "zh-cn", "zh-hans", "cn", "simplified", "simplified-chinese"}:
        return "zh"
    return "en"


def language_instruction(language: str) -> str:
    if normalize_language(language) == "zh":
        return "Use 简体中文 for the entire report."
    return "Use English for the entire report."


def format_coffee_list(
    products: list[ProductCandidate],
    page_text_by_id: dict[str, str],
    max_chars: int,
) -> str:
    if not products:
        return ""
    lines = ["## Coffee list", ""]
    for product in products:
        fields = sanitize_product_fields(product)
        body_text = ""
        if product.body_html:
            body_text = sanitize_html_to_text(
                product.body_html, max_chars, remove_boilerplate=False
            )
        body_text = sanitize_prompt_field(body_text, max_chars)
        page_text = sanitize_prompt_field(
            page_text_by_id.get(product.product_id, ""), max_chars
        )
        description_text = page_text or body_text
        entry_lines = [
            f"- product_id: {fields['product_id']}",
            f"  name: {fields['name']}",
            f"  url: {fields['url']}",
        ]
        if fields["list_price"]:
            entry_lines.append(f"  list price: {fields['list_price']}")
        if fields["list_badge"]:
            entry_lines.append(f"  badge: {fields['list_badge']}")
        entry_lines.extend(format_variant_lines(product.variants))
        if description_text:
            entry_lines.append("  description:")
            entry_lines.append(f"  {description_text}")
        entry_lines.append("")
        lines.append("\n".join(entry_lines))
    return "\n".join(lines) + "\n"


def sanitize_product_fields(product: ProductCandidate) -> dict[str, str]:
    return {
        "product_id": sanitize_prompt_field(product.product_id, 200),
        "name": sanitize_prompt_field(product.name, 200),
        "url": sanitize_prompt_field(product.url, 400),
        "list_price": sanitize_prompt_field(product.list_price, 40),
        "list_badge": sanitize_prompt_field(product.list_badge, 40),
    }


def build_batch_prompt(
    roaster_name: str,
    products: list[ProductCandidate],
    page_text_by_id: dict[str, str],
    max_chars: int,
    language: str,
) -> str:
    header = (
        f"You are evaluating coffees from {sanitize_prompt_field(roaster_name, 200)}. "
        "Your goal is to find the best coffees available right now.\n"
        "Do not review every item; focus only on standout coffees and ignore routine "
        "offerings unless there is clear evidence of exceptional quality.\n"
        "High-end signals: rare varieties (e.g., Geisha, Sudan Rume, SL28), exceptional "
        "producers or farms, competition lots, Cup of Excellence winners, limited microlots, "
        "experimental processing (e.g., anaerobic, thermal shock), unusually high cupping "
        "scores, or strong peer reputation. Use clear evidence from the page text or "
        "grounded sources.\n"
        "Think carefully and generate a coherent, complete markdown recommendation report.\n"
        "You are free to choose your own structure; avoid empty placeholder sections.\n"
        "For each recommendation, explain why it is exceptional and cite the specific "
        "signals from the provided text. It is OK to recommend nothing if nothing stands "
        "out; say so explicitly.\n"
        f"{language_instruction(language)}\n\n"
        "Products:\n"
    )
    sections: list[str] = [header]
    for product in products:
        fields = sanitize_product_fields(product)
        body_text = ""
        if product.body_html:
            body_text = sanitize_html_to_text(
                product.body_html, max_chars, remove_boilerplate=False
            )
        body_text = sanitize_prompt_field(body_text, max_chars)
        page_text = sanitize_prompt_field(
            page_text_by_id.get(product.product_id, ""), max_chars
        )
        description_text = page_text or body_text
        description_block = (
            f"  description:\n  {description_text}" if description_text else ""
        )
        lines = [
            f"- product_id: {fields['product_id']}",
            f"  name: {fields['name']}",
            f"  url: {fields['url']}",
        ]
        if fields["list_price"]:
            lines.append(f"  list price: {fields['list_price']}")
        if fields["list_badge"]:
            lines.append(f"  badge: {fields['list_badge']}")
        lines.extend(format_variant_lines(product.variants))
        if description_block:
            lines.append(description_block)
        lines.append("")
        sections.append("\n".join(lines))
    return "\n".join(sections)


def build_digest_prompt(
    reports: list[tuple[str, str]],
    language: str,
) -> str:
    header = (
        "You are given markdown reports for multiple coffee roasters.\n"
        "Write a digest that synthesizes the key recommendations across all reports.\n"
        "Be detailed in your recommendations and reasoning, and explain unfamiliar terms.\n"
        "Do not worry about a long report length.\n"
        "Include: overall summary, standout coffees and why, any roasters with no strong picks, "
        "and final overall recommendations.\n"
        "In the summary, explicitly list all roasters represented in the reports; "
        "do not assume a fixed set.\n"
        "Only use the information provided in the reports; do not introduce new coffees "
        "or roasters.\n"
        f"{language_instruction(language)}\n\n"
    )
    sections = [header]
    for name, text in reports:
        sections.append(f"## Report: {name}\n\n{text}\n")
    return "\n".join(sections)


def build_new_products_digest_prompt(
    items: list[dict[str, Any]],
    max_chars: int,
    language: str,
) -> str:
    header = (
        "You are given a list of newly discovered coffees from the past 7 days across multiple roasters.\n"
        "Write a digest of the best new coffees from this 7-day window.\n"
        "Be detailed in your recommendations and reasoning, and explain unfamiliar terms.\n"
        "Do not worry about a long report length.\n"
        "Include: overall summary, standout coffees and why, any roasters with no strong picks, "
        "and final overall recommendations.\n"
        "Only use the information provided below; do not introduce new coffees.\n"
        f"{language_instruction(language)}\n\n"
        "New coffees:\n"
    )
    sections: list[str] = [header]
    for item in items:
        roaster = sanitize_prompt_field(str(item.get("roaster", "")), 200)
        product_id = sanitize_prompt_field(str(item.get("product_id", "")), 200)
        name = sanitize_prompt_field(str(item.get("name", "")), 200)
        url = sanitize_prompt_field(str(item.get("url", "")), 400)
        list_price = sanitize_prompt_field(str(item.get("list_price", "")), 40)
        badge = sanitize_prompt_field(str(item.get("badge", "")), 40)
        description = sanitize_prompt_field(str(item.get("description", "")), max_chars)
        variant_lines = item.get("variant_lines")
        lines = [
            f"- roaster: {roaster}",
            f"  product_id: {product_id}",
            f"  name: {name}",
            f"  url: {url}",
        ]
        if list_price:
            lines.append(f"  list price: {list_price}")
        if badge:
            lines.append(f"  badge: {badge}")
        if isinstance(variant_lines, (list, tuple)):
            lines.extend([line for line in variant_lines if isinstance(line, str)])
        if description:
            lines.append("  description:")
            lines.append(f"  {description}")
        lines.append("")
        sections.append("\n".join(lines))
    return "\n".join(sections)


def build_roaster_ratings_digest_prompt(
    reports: list[tuple[str, str]],
    language: str,
) -> str:
    header = (
        "You are given markdown reports for multiple coffee roasters.\n"
        "Rate each roaster's current offerings based on the strength of standout coffees "
        "in its report. Use a 1-10 score where 10 means an exceptional lineup right now "
        "and 1 means no compelling coffees.\n"
        "Provide a scorecard that lists every roaster and its rating, then detailed analysis "
        "per roaster.\n"
        "Recommend highlight roasters (strongest current lineups) and lowlight roasters "
        "(weakest current lineups), with detailed reasoning and background.\n"
        "Do not worry about a long report length.\n"
        "Explicitly list all roasters represented in the reports; do not assume a fixed set.\n"
        "Only use the information provided in the reports; do not introduce new coffees "
        "or roasters.\n"
        f"{language_instruction(language)}\n\n"
    )
    sections = [header]
    for name, text in reports:
        sections.append(f"## Report: {name}\n\n{text}\n")
    return "\n".join(sections)
