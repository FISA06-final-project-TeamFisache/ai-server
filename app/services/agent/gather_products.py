"""
모으기 계좌 추천용 정적 상품 데이터.

사용자가 해당 유형의 계좌를 보유하지 않을 때 asset_portfolio 에이전트가
신규 개설 계좌를 추천하는 데만 사용한다.
필드: product_type, institution, name, interest_rate
"""

GATHER_PRODUCTS: list[dict] = [
    {
        "product_type": "DEPOSIT",
        "institution": "우리은행",
        "name": "WON플러스예금",
        "interest_rate": 2.15,
    },
    {
        "product_type": "SAVING",
        "institution": "우리은행",
        "name": "WON적금",
        "interest_rate": 3.15,
    },
    {
        "product_type": "ISA",
        "institution": "우리투자증권",
        "name": "우리투자증권 중개형 ISA",
        "interest_rate": None,
    },
    {
        "product_type": "IRP",
        "institution": "우리투자증권",
        "name": "우리투자증권 개인형 IRP",
        "interest_rate": None,
    },
    {
        "product_type": "PENSION_SAVINGS",
        "institution": "우리투자증권",
        "name": "우리투자증권 연금저축계좌",
        "interest_rate": None,
    },
]
