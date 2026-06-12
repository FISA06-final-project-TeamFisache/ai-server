import logging
from datetime import datetime, timezone
from uuid import UUID

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from app.schemas.salary import SalaryRequest, SalaryResponse, PortfolioItem, FlowItem
from app.services.agent.llm import ainvoke_structured

logger = logging.getLogger(__name__)


class _ItemRatio(BaseModel):
    asset_id: str
    ratio: float  # 0.0 ~ 1.0, 전체 합계 = 1.0

class _RatioOutput(BaseModel):
    item_ratios: list[_ItemRatio]

class _CommentOnly(BaseModel):
    rebalance_comment: str


def _apply_ratios(
    portfolio: list[dict],
    flows: list[dict],
    ratio_map: dict[str, float],
    salary_diff: int,
) -> tuple[list[dict], list[dict]]:
    """ratio × salary_diff를 각 항목에 개별 적용. 반올림 오차는 최대 비율 항목에 흡수."""
    all_items = portfolio + flows
    deltas: dict[str, int] = {v["asset_id"]: round(ratio_map.get(v["asset_id"], 0.0) * salary_diff) for v in all_items}

    remainder = salary_diff - sum(deltas.values())
    if remainder != 0 and ratio_map:
        top_id = max(ratio_map, key=lambda k: ratio_map[k])
        deltas[top_id] = deltas.get(top_id, 0) + remainder

    # 결손금: 잔액 초과 차감 방지 — 미처리 결손금은 잔액 있는 항목에 순차 재분배
    if salary_diff < 0:
        amount_by_id = {v["asset_id"]: v["amount"] for v in all_items}
        overflow = 0
        for aid in list(deltas):
            if amount_by_id[aid] + deltas[aid] < 0:
                overflow += -(amount_by_id[aid] + deltas[aid])
                deltas[aid] = -amount_by_id[aid]

        if overflow > 0:
            for v in sorted(all_items, key=lambda x: -(amount_by_id[x["asset_id"]] + deltas.get(x["asset_id"], 0))):
                if overflow <= 0:
                    break
                aid = v["asset_id"]
                available = amount_by_id[aid] + deltas.get(aid, 0)
                if available <= 0:
                    continue
                take = min(overflow, available)
                deltas[aid] -= take
                overflow -= take

    def apply(items: list[dict]) -> list[dict]:
        return [{**v, "amount": max(0, v["amount"] + deltas.get(v["asset_id"], 0))} for v in items]

    return apply(portfolio), apply(flows)


async def _generate_comment(
    salary_diff: int,
    all_items: list[dict],
    deltas: dict[str, int],
    expense_summary: str,
) -> str:
    """실제 계산된 delta 금액을 기반으로 코멘트를 생성한다."""
    direction = "잉여금" if salary_diff > 0 else "결손금"

    changed_lines = []
    unchanged_names = []
    for item in all_items:
        aid = item["asset_id"]
        name = item.get("account_purpose") or item.get("title", aid)
        delta = deltas.get(aid, 0)
        if delta != 0:
            changed_lines.append(f"  - {name}: {delta:+,}원")
        else:
            unchanged_names.append(name)

    changed_desc = "\n".join(changed_lines) or "  없음"
    unchanged_desc = ", ".join(unchanged_names) or "없음"

    messages = [
        SystemMessage(content=(
            "당신은 월급 배분 전문가입니다. 아래 실제 조정 결과를 바탕으로 사용자에게 설명하는 코멘트를 2~3문장으로 작성하세요.\n\n"
            "규칙:\n"
            "- 말투는 반드시 '~습니다' 체를 사용하세요.\n"
            "- 결손금/잉여금은 월급 변동에 의한 것입니다. '소비 때문에 결손금/잉여금이 발생했다'는 표현은 절대 금지입니다.\n"
            "- 각 항목에 얼마를 넣었는지·뺐는지 조정 결과 금액은 쓰지 마세요. 그 숫자는 아래 화면에 이미 표시됩니다.\n"
            "- 소비 카테고리 금액(예: 식비 630,000원)을 반드시 근거로 언급하세요.\n"
            "- 투자 관련 수치(수익률, 비중 등)도 구체적으로 언급해도 좋습니다.\n"
            "- 어떤 소비 패턴을 참고했는지, 그래서 어떤 항목을 늘리거나 줄였는지, 그 결과 어떤 효과를 기대하는지 중심으로 서술하세요.\n"
            "- '변동 없는 항목'은 설명하지 마세요.\n"
            "- 내부 용어(ratio, asset_id, portfolio_item, flow_item 등) 사용 금지.\n"
            "- 반드시 아래 JSON 형식으로만 응답하세요:\n"
            '{"rebalance_comment": "..."}'
        )),
        HumanMessage(content=(
            f"월급 {direction}: {salary_diff:+,}원\n\n"
            f"이번 달 소비 패턴:\n{expense_summary}\n\n"
            f"조정된 항목:\n{changed_desc}\n\n"
            f"변동 없는 항목 (언급하지 마세요): {unchanged_desc}"
        )),
    ]

    result = await ainvoke_structured(messages, _CommentOnly)
    if result:
        return result.rebalance_comment
    return f"월급 {direction} {abs(salary_diff):,}원을 반영하여 각 항목을 조정했습니다."


async def analyze_salary_rebalance(request: SalaryRequest) -> SalaryResponse:
    portfolio_current = [
        {"asset_id": str(p.asset_id), "account_purpose": p.account_purpose, "amount": p.amount}
        for p in request.portfolio_items
    ]
    flow_current = [
        {"asset_id": str(f.asset_id), "title": f.title, "term": f.term,
         "summary": f.summary, "amount": f.amount}
        for f in request.flow_items
    ]
    all_items = portfolio_current + flow_current

    expense_summary = "\n".join(
        f"  - {e.name}: {e.expense:,}원" for e in request.category_expense
    ) or "  소비 내역 없음"

    portfolio_desc = "\n".join(
        f"  - [asset_id: {p['asset_id']}] {p['account_purpose']}: {p['amount']:,}원"
        for p in portfolio_current
    ) or "  없음"

    flow_desc = "\n".join(
        f"  - [asset_id: {f['asset_id']}] {f['title']}({f['term']}), {f['summary']}: {f['amount']:,}원"
        for f in flow_current
    ) or "  없음"

    direction = "잉여금" if request.salary_diff > 0 else "결손금"
    all_ids = [v["asset_id"] for v in all_items]

    # ── Step 1: ratio만 결정 ──────────────────────────────────
    ratio_messages = [
        SystemMessage(content=(
            "당신은 월급 배분 전문가입니다.\n"
            "월급 변동액을 각 항목의 성격과 소비 패턴을 고려해 항목별 비율로 배분하세요.\n\n"
            "판단 기준:\n"
            "- 잉여금(+): 투자 포트폴리오(flow_items) 비중을 높이는 방향을 우선 검토하세요.\n"
            "- 결손금(-): 생활비·비상금 같은 필수 항목보다 저축·투자 항목을 먼저 줄이세요.\n"
            "- account_purpose, title, summary, term을 참고해 맥락에 맞게 판단하세요.\n"
            "- 소비 지출이 많은 달(식비·쇼핑 등 증가)이면 생활비 비율을 높이고 투자 비율을 낮추세요.\n"
            "- 소비가 적은 달이면 투자 비율을 더 높여도 좋습니다.\n"
            "- 특정 카테고리 지출이 급증한 경우 해당 목적의 항목 비율을 우선 보호하세요.\n\n"
            "출력 규칙:\n"
            "- item_ratios에 제공된 모든 asset_id를 빠짐없이 포함하세요.\n"
            "- ratio는 잉여금/결손금 여부에 관계없이 항상 0.0~1.0 사이의 양수입니다.\n"
            "- ratio는 '이 항목이 전체 변동액 중 몇 %를 담당하는가'를 나타냅니다.\n"
            "  - 잉여금(+): ratio > 0인 항목에 돈이 추가됩니다.\n"
            "  - 결손금(-): ratio > 0인 항목에서 돈이 차감됩니다. 절대 음수로 쓰지 마세요.\n"
            "- 모든 ratio의 합계는 반드시 정확히 1.0이어야 합니다.\n"
            "- 변동시키지 않을 항목은 ratio를 0.0으로 설정하세요.\n\n"
            "반드시 아래 JSON만 응답하세요:\n"
            '{"item_ratios": [{"asset_id": "...", "ratio": 0.0}, ...]}'
        )),
        HumanMessage(content=(
            f"월급 {direction}: {request.salary_diff:+,}원\n\n"
            f"이번 달 소비 패턴:\n{expense_summary}\n\n"
            f"portfolio_items (생활비·저축·비상금 등):\n{portfolio_desc}\n\n"
            f"flow_items (투자 포트폴리오):\n{flow_desc}\n\n"
            f"배분 대상 asset_id 목록: {all_ids}"
        )),
    ]

    result = await ainvoke_structured(ratio_messages, _RatioOutput)

    if result is None:
        valid = False
        logger.warning("[salary_rebalance] LLM 응답 None — 폴백 적용")
    else:
        expected_ids = {v["asset_id"] for v in all_items}
        got_ids = {r.asset_id for r in result.item_ratios}
        ratio_sum = sum(r.ratio for r in result.item_ratios)
        negative = [r for r in result.item_ratios if r.ratio < 0.0]
        valid = (
            got_ids == expected_ids
            and not negative
            and abs(ratio_sum - 1.0) < 0.05
        )
        if not valid:
            logger.warning(
                "[salary_rebalance] LLM 응답 불일치 — 폴백 적용 "
                "(missing_ids=%s, extra_ids=%s, negative_ratios=%s, ratio_sum=%.4f)",
                expected_ids - got_ids,
                got_ids - expected_ids,
                [(r.asset_id, r.ratio) for r in negative],
                ratio_sum,
            )

    if valid:
        total = sum(r.ratio for r in result.item_ratios)
        ratio_map = {r.asset_id: r.ratio / total for r in result.item_ratios}
    else:
        # 폴백: 잉여금·결손금 모두 flow(투자) 우선
        targets = flow_current if flow_current else portfolio_current
        if not targets:
            ratio_map = {}
        else:
            fallback_ids = {v["asset_id"] for v in targets}
            ratio_map = {
                v["asset_id"]: (1.0 / len(targets) if v["asset_id"] in fallback_ids else 0.0)
                for v in all_items
            }

    # ── Step 2: 실제 금액 계산 ────────────────────────────────
    adj_portfolio, adj_flows = _apply_ratios(portfolio_current, flow_current, ratio_map, request.salary_diff)

    deltas: dict[str, int] = {}
    for cur, adj in zip(portfolio_current, adj_portfolio):
        deltas[cur["asset_id"]] = adj["amount"] - cur["amount"]
    for cur, adj in zip(flow_current, adj_flows):
        deltas[cur["asset_id"]] = adj["amount"] - cur["amount"]

    # ── Step 3: 실제 숫자 기반으로 코멘트 생성 ───────────────
    comment = await _generate_comment(request.salary_diff, all_items, deltas, expense_summary)

    return SalaryResponse(
        created_at=datetime.now(timezone.utc),
        portfolio_items=[
            PortfolioItem(asset_id=UUID(a["asset_id"]), account_purpose=a["account_purpose"], amount=a["amount"])
            for a in adj_portfolio
        ],
        flow_items=[
            FlowItem(asset_id=UUID(f["asset_id"]), title=f["title"], term=f["term"],
                     summary=f["summary"], amount=f["amount"])
            for f in adj_flows
        ],
        rebalance_comment=comment,
    )
