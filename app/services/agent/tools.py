import json
import logging

from langchain_core.tools import tool

logger = logging.getLogger(__name__)


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


@tool
def calculate_hrp_weights(returns_json: str) -> str:
    """
    수익률 시계열 JSON → HRP(계층적 리스크 패리티) 최적 비중 계산.
    입력: pandas DataFrame.to_json() 형식 (컬럼=ETF명, 행=날짜별 수익률)
    반환: {"ETF명": 비중, ...} JSON. 비중 합계는 수학적으로 1.0 보장.
    데이터 부족(< 20일) 또는 상품 1개이면 균등/단독 가중치로 폴백.
    """
    try:
        import pandas as pd
        from pypfopt.hierarchical_portfolio import HRPOpt

        returns = pd.read_json(returns_json)
        if returns.empty:
            return json.dumps({"error": "빈 데이터"})

        if len(returns.columns) == 1:
            return json.dumps({returns.columns[0]: 1.0})

        if len(returns) < 20:
            n = len(returns.columns)
            return json.dumps({c: round(1.0 / n, 4) for c in returns.columns})

        hrp = HRPOpt(returns)
        weights = hrp.optimize()
        return json.dumps({k: round(v, 4) for k, v in weights.items()})

    except ImportError:
        logger.warning("PyPortfolioOpt 미설치 — 균등 가중치로 폴백")
    except Exception as e:
        logger.warning("HRP 계산 실패: %s", e)

    # 폴백: 균등 가중치
    try:
        import pandas as pd
        cols = pd.read_json(returns_json).columns.tolist()
        eq = round(1.0 / len(cols), 4) if cols else 1.0
        return json.dumps({c: eq for c in cols})
    except Exception:
        return json.dumps({"error": "계산 실패"})


FINANCE_TOOLS = [compound_interest, monthly_savings_needed, normalize_ratios, rebalance_diff, calculate_hrp_weights]
