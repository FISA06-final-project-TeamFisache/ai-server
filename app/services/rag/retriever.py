"""products 테이블에서 금융 상품 정보를 조회해 LLM 컨텍스트 문자열로 변환한다."""
from __future__ import annotations

import logging

from app.services.rag.db import get_pool
from app.services.rag.knowledge import FINANCIAL_KNOWLEDGE

logger = logging.getLogger(__name__)

_TYPE_LABELS = {
    "STOCK": "주식/ETF",
    "BOND": "채권",
    "DEPOSIT": "예금",
    "SAVING": "적금",
    "IRP": "IRP(개인형퇴직연금)",
}

_ALIAS_MAP = {
    "stock": "STOCK",
    "bond": "BOND",
    "cash": "DEPOSIT",
    "deposit": "DEPOSIT",
    "saving": "SAVING",
    "irp": "IRP",
    "주식": "STOCK",
    "채권": "BOND",
    "현금": "DEPOSIT",
    "예금": "DEPOSIT",
    "적금": "SAVING",
}


async def get_products_context(
    product_types: list[str] | None = None,
    limit_per_type: int = 3,
) -> str:
    """
    DB에서 상품을 조회해 프롬프트용 문자열로 반환한다.
    DB 연결이 없거나 실패하면 빈 문자열을 반환해 에이전트가 계속 동작하도록 한다.
    """
    pool = await get_pool()
    if pool is None:
        return ""

    try:
        if product_types:
            normalized = list({
                _ALIAS_MAP.get(t.lower(), t.upper()) for t in product_types
            })
            rows = await pool.fetch(
                "SELECT product_type, institution, name, interest_rate, description "
                "FROM products "
                "WHERE product_type = ANY($1::text[]) AND deleted_at IS NULL "
                "ORDER BY product_type, interest_rate DESC NULLS LAST",
                normalized,
            )
        else:
            rows = await pool.fetch(
                "SELECT product_type, institution, name, interest_rate, description "
                "FROM products "
                "WHERE deleted_at IS NULL "
                "ORDER BY product_type, interest_rate DESC NULLS LAST "
                "LIMIT $1",
                limit_per_type * len(_TYPE_LABELS),
            )

        if not rows:
            return ""

        grouped: dict[str, list] = {}
        for row in rows:
            grouped.setdefault(row["product_type"], []).append(row)

        lines = ["[우리은행 금융 상품 현황]"]
        for pt, prows in grouped.items():
            label = _TYPE_LABELS.get(pt, pt)
            lines.append(f"\n■ {label}")
            for row in prows[:limit_per_type]:
                rate_str = f"{row['interest_rate']}%" if row["interest_rate"] else "금리 별도"
                desc = (row["description"] or "")[:80].replace("\n", " ")
                lines.append(f"  · {row['institution']} {row['name']} ({rate_str}) — {desc}")

        return "\n".join(lines)

    except Exception as e:
        logger.warning("상품 조회 실패: %s", e)
        return ""


async def get_products_list(
    product_types: list[str] | None = None,
    limit: int = 30,
) -> list[dict]:
    """DB에서 상품을 조회해 dict 리스트로 반환한다."""
    pool = await get_pool()
    if pool is None:
        return []

    try:
        if product_types:
            normalized = list({
                _ALIAS_MAP.get(t.lower(), t.upper()) for t in product_types
            })
            rows = await pool.fetch(
                "SELECT product_type, institution, name, interest_rate, description "
                "FROM products "
                "WHERE product_type = ANY($1::text[]) AND deleted_at IS NULL "
                "ORDER BY product_type, interest_rate DESC NULLS LAST "
                "LIMIT $2",
                normalized, limit,
            )
        else:
            rows = await pool.fetch(
                "SELECT product_type, institution, name, interest_rate, description "
                "FROM products "
                "WHERE deleted_at IS NULL "
                "ORDER BY product_type, interest_rate DESC NULLS LAST "
                "LIMIT $1",
                limit,
            )
        return [dict(row) for row in rows]
    except Exception as e:
        logger.warning("상품 목록 조회 실패: %s", e)
        return []


async def build_rag_context(product_types: list[str] | None = None) -> str:
    """금융 지식 + DB 상품 정보를 합쳐 단일 컨텍스트 문자열을 반환한다."""
    products = await get_products_context(product_types)
    parts = [FINANCIAL_KNOWLEDGE]
    if products:
        parts.append(products)
    return "\n\n".join(parts)
