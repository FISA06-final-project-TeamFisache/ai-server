from langchain_core.tools import tool


@tool
def compound_interest(principal: float, monthly_rate: float, months: int) -> float:
    """복리 계산. 원금(principal), 월이율(monthly_rate), 기간(months) → 최종 금액."""
    return round(principal * (1 + monthly_rate) ** months, 2)


@tool
def monthly_savings_needed(target: float, current: float, months: int, monthly_rate: float = 0.003) -> float:
    """목표 금액 달성에 필요한 월 저축액 계산.
    target: 목표 금액, current: 현재 보유액, months: 남은 개월 수, monthly_rate: 월 수익률(기본 0.3%)
    """
    if months <= 0:
        return max(0.0, target - current)
    future_value_of_current = current * (1 + monthly_rate) ** months
    remaining = target - future_value_of_current
    if remaining <= 0:
        return 0.0
    if monthly_rate == 0:
        return round(remaining / months, 2)
    return round(remaining * monthly_rate / ((1 + monthly_rate) ** months - 1), 2)


@tool
def normalize_ratios(stock: float, bond: float, cash: float) -> dict:
    """주식·채권·현금 비율을 합계 100%로 정규화.
    각 값이 0 이상이어야 하며, 소수점 이하는 반올림 처리됩니다.
    """
    total = stock + bond + cash
    if total == 0:
        return {"stock_ratio": 33, "bond_ratio": 33, "cash_ratio": 34}
    s = round(stock / total * 100)
    b = round(bond / total * 100)
    c = 100 - s - b
    return {"stock_ratio": s, "bond_ratio": b, "cash_ratio": c}


@tool
def rebalance_diff(current_ratio: float, target_ratio: float, total_asset: float) -> dict:
    """현재 비율과 목표 비율의 차이를 금액으로 환산.
    current_ratio, target_ratio: 0~100 사이 퍼센트 값
    total_asset: 총 자산 금액
    """
    current_amount = total_asset * current_ratio / 100
    target_amount = total_asset * target_ratio / 100
    diff = target_amount - current_amount
    return {
        "current_amount": round(current_amount, 2),
        "target_amount": round(target_amount, 2),
        "diff": round(diff, 2),
        "action": "매수" if diff > 0 else "매도" if diff < 0 else "유지",
    }


FINANCE_TOOLS = [compound_interest, monthly_savings_needed, normalize_ratios, rebalance_diff]
