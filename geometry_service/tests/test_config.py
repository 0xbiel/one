import os
import unittest
from unittest.mock import patch

from geometry_service.config import ServiceSettings


class ServiceSettingsTests(unittest.TestCase):
    def test_positioning_workers_default_and_bounds(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(ServiceSettings.from_env().positioning_workers, 3)

        with patch.dict(os.environ, {"ONE_POSITIONING_WORKERS": "0"}, clear=True):
            self.assertEqual(ServiceSettings.from_env().positioning_workers, 1)

        with patch.dict(os.environ, {"ONE_POSITIONING_WORKERS": "99"}, clear=True):
            self.assertEqual(ServiceSettings.from_env().positioning_workers, 8)

        with patch.dict(os.environ, {"ONE_POSITIONING_WORKERS": "invalid"}, clear=True):
            self.assertEqual(ServiceSettings.from_env().positioning_workers, 3)


if __name__ == "__main__":
    unittest.main()
