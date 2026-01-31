from __future__ import annotations

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


def format_coffee_list(products: list[ProductCandidate]) -> str:
    if not products:
        return ""
    lines = ["## Coffee list", ""]
    for product in products:
        name = product.name.strip() or "Unknown"
        url = product.url.strip()
        if url:
            lines.append(f"- {name} ({url})")
        else:
            lines.append(f"- {name}")
    return "\n".join(lines) + "\n"


def sanitize_product_fields(product: ProductCandidate) -> dict[str, str]:
    return {
        "product_id": sanitize_prompt_field(product.product_id, 200),
        "name": sanitize_prompt_field(product.name, 200),
        "url": sanitize_prompt_field(product.url, 400),
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
        "Decide if each is worth trying.\n"
        "These roasters are already high-end; only recommend coffees that are exceptional "
        "even by specialty standards. Treat routine offerings as not worth trying unless "
        "there is clear evidence of standout quality.\n"
        "High-end signals: rare varieties (e.g., Geisha, Sudan Rume, SL28), exceptional "
        "producers or farms, competition lots, Cup of Excellence winners, limited microlots, "
        "experimental processing (e.g., anaerobic, thermal shock), unusually high cupping "
        "scores, or strong peer reputation. Use clear evidence from the page text or "
        "grounded sources.\n"
        "Think carefully and generate a coherent, complete markdown recommendation report.\n"
        "You are free to choose your own structure; avoid empty placeholder sections.\n"
        "Elaborate on the reasons for each recommendation.\n"
        "It is OK to recommend nothing if nothing stands out; say so explicitly.\n"
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
        description_block = (
            f"  description:\n  {body_text}" if body_text else "  description: (none available)"
        )
        page_text_block = ""
        if page_text and page_text != body_text:
            page_text_block = f"  page text:\n  {page_text}"
        sections.append(
            "\n".join(
                [
                    f"- product_id: {fields['product_id']}",
                    f"  name: {fields['name']}",
                    f"  url: {fields['url']}",
                    *format_variant_lines(product.variants),
                    description_block,
                    page_text_block,
                    "",
                ]
            )
        )
    return "\n".join(sections)


def build_digest_prompt(
    reports: list[tuple[str, str]],
    language: str,
) -> str:
    header = (
        "You are given markdown reports for multiple coffee roasters.\n"
        "Write a concise digest that synthesizes the key recommendations across all reports.\n"
        "Include: overall summary, standout coffees and why, any roasters with no strong picks, "
        "and final overall recommendations.\n"
        "Only use the information provided in the reports; do not introduce new coffees.\n"
        f"{language_instruction(language)}\n\n"
    )
    sections = [header]
    for name, text in reports:
        sections.append(f"## Report: {name}\n\n{text}\n")
    return "\n".join(sections)
