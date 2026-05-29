from datetime import date


class DailyCache:
    def __init__(self) -> None:
        self._ref: date = date.today()
        self._totals: dict[str, int] = {}
        self._by_cat: dict[str, dict[str, int]] = {}
        self._alerted: dict[str, date] = {}
        self._db: dict[str, tuple[dict[str, int], dict[str, int], date]] = {}

    def reset_if_new_day(self, today: date) -> None:
        if today != self._ref:
            self._ref = today
            self._totals.clear()
            self._by_cat.clear()

    def accumulate(self, asset_number: str, amount: int, category: str) -> None:
        self._totals[asset_number] = self._totals.get(asset_number, 0) + amount
        cat_map = self._by_cat.setdefault(asset_number, {})
        cat_map[category] = cat_map.get(category, 0) + amount

    def get_today_total(self, asset_number: str) -> int:
        return self._totals.get(asset_number, 0)

    def get_today_by_category(self, asset_number: str) -> dict[str, int]:
        return dict(sorted(self._by_cat.get(asset_number, {}).items(), key=lambda x: -x[1]))

    def is_alerted(self, asset_number: str, today: date) -> bool:
        return self._alerted.get(asset_number) == today

    def mark_alerted(self, asset_number: str, today: date) -> None:
        self._alerted[asset_number] = today

    def get_db_cache(self, asset_number: str, today: date) -> tuple[dict[str, int], dict[str, int]] | None:
        entry = self._db.get(asset_number)
        if entry and entry[2] == today:
            return entry[0], entry[1]
        return None

    def set_db_cache(
        self,
        asset_number: str,
        db_this_by_cat: dict[str, int],
        last_by_cat: dict[str, int],
        today: date,
    ) -> None:
        self._db[asset_number] = (db_this_by_cat, last_by_cat, today)


daily_cache = DailyCache()
