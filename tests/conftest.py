import pytest
from unittest.mock import patch, MagicMock

from config import settings


@pytest.fixture(autouse=True)
def mock_settings_validate():
    with patch.object(settings, "validate"):
        yield