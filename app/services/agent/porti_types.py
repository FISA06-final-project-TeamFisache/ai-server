"""Spring PortiType enum 매핑 — 모든 agent에서 공유"""

PORTI_TYPE_DESC: dict[str, str] = {
    "SWIMMING": "안전형_단기형 — 원금 보전 선호, 단기 투자",
    "ARCHERY":  "안전형_장기형 — 원금 보전 선호, 장기 투자",
    "JUDO":     "중립형_단기형 — 균형 투자 성향, 단기 투자",
    "RHYTHMIC": "중립형_장기형 — 균형 투자 성향, 장기 투자",
    "FENCING":  "투자형_단기형 — 수익 추구 성향, 단기 투자",
    "CYCLING":  "투자형_장기형 — 수익 추구 성향, 장기 투자",
}

# 배분 전략 가이드 — 각 성향에 맞는 저축·투자 우선순위 설명
PORTI_GUIDANCE: dict[str, str] = {
    "SWIMMING": (
        "원금 손실을 극도로 꺼리며 단기(1년 이내) 목표를 우선합니다. "
        "비상금과 예비비를 최대한 확보한 뒤 나머지를 안정적인 예적금에 집중하세요. "
        "투자 비중은 최소화하고, 가처분소득의 대부분을 유동성 높은 계좌에 배분하세요."
    ),
    "ARCHERY": (
        "원금 손실을 꺼리지만 장기(3년 이상) 관점으로 자산을 불려 나갑니다. "
        "비상금을 충분히 확보한 뒤 남은 여력은 장기 정기예금에 꾸준히 적립하세요. "
        "투자 비중은 최소로 유지하되, 장기 복리 저축을 핵심 전략으로 삼으세요."
    ),
    "JUDO": (
        "안정성과 수익성을 균형 있게 추구하며 단기(1~2년) 목표를 병행합니다. "
        "비상금을 마련한 뒤 저축과 투자를 대략 6:4 비율로 배분하세요. "
        "단기 내 사용할 여유 자금도 파킹통장에 별도로 확보해두세요."
    ),
    "RHYTHMIC": (
        "안정성과 수익성을 균형 있게 추구하며 장기(3년 이상) 관점을 유지합니다. "
        "비상금을 충분히 마련한 뒤 중장기 적립식 저축과 투자를 고르게 배분하세요. "
        "꾸준한 투자 습관을 통해 장기 복리 효과를 극대화하는 방향으로 설계하세요."
    ),
    "FENCING": (
        "손실 위험을 감수하더라도 적극적 수익을 추구하며 단기(1년 이내) 목표를 지향합니다. "
        "생활비와 최소한의 비상금만 확보하고, 남은 여력의 상당 부분을 투자에 집중하세요. "
        "투자 비중을 높이되 단기 유동성 확보를 위해 일부는 파킹통장에 유지하세요."
    ),
    "CYCLING": (
        "손실 위험을 감수하더라도 적극적 수익을 추구하며 장기(3년 이상) 복리를 지향합니다. "
        "생활비와 최소한의 비상금만 확보하고, 남은 여력의 대부분을 장기 투자에 집중하세요. "
        "복리 효과를 최대화하는 장기 관점에서 투자 비중을 가장 높게 설정하세요."
    ),
}

# 안전형 유형 집합 (계좌 배정·투자 비중 결정에 사용)
STABLE_PORTI_TYPES: frozenset[str] = frozenset({"SWIMMING", "ARCHERY"})
NEUTRAL_PORTI_TYPES: frozenset[str] = frozenset({"JUDO", "RHYTHMIC"})
INVEST_PORTI_TYPES: frozenset[str] = frozenset({"FENCING", "CYCLING"})


def porti_label(porti_type: str) -> str:
    """LLM 프롬프트용 — 코드와 한국어 설명을 함께 반환.
    예: 'CYCLING (투자형_장기형 — 수익 추구 성향, 장기 투자)'
    """
    return f"{porti_type} ({PORTI_TYPE_DESC.get(porti_type, porti_type)})"


def porti_detail(porti_type: str) -> str:
    """LLM 프롬프트용 — 성향 레이블과 배분 전략 가이드를 함께 반환."""
    label = porti_label(porti_type)
    guidance = PORTI_GUIDANCE.get(porti_type, "")
    return f"{label}\n  배분 전략: {guidance}" if guidance else label
