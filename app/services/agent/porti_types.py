"""Spring PortiType enum 매핑 — 모든 agent에서 공유"""

PORTI_TYPE_DESC: dict[str, str] = {
    "SWIMMING": "안전형_단기형 — 원금 보전 선호, 단기 투자",
    "ARCHERY":  "안전형_장기형 — 원금 보전 선호, 장기 투자",
    "JUDO":     "중립형_단기형 — 균형 투자 성향, 단기 투자",
    "RHYTHMIC": "중립형_장기형 — 균형 투자 성향, 장기 투자",
    "FENCING":  "투자형_단기형 — 수익 추구 성향, 단기 투자",
    "CYCLING":  "투자형_장기형 — 수익 추구 성향, 장기 투자",
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
