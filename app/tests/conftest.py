import asyncio
import sys

import pytest
from fastapi.testclient import TestClient

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


@pytest.fixture
def client():
    from app.main import app
    with TestClient(app) as c:
        yield c
