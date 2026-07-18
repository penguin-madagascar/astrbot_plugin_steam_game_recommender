from __future__ import annotations

import logging
import re
import unittest
from unittest.mock import Mock

from astrbot_plugin_steam_game_recommender.services.safe_errors import (
    log_external_failure,
    safe_error_id,
)


class SafeErrorsTest(unittest.TestCase):
    def test_external_failure_log_never_includes_exception_text(self) -> None:
        logger = logging.getLogger("safe-errors-test")
        secret = "secret /private/provider/path?token=abcdef"

        with self.assertLogs(logger, level="WARNING") as captured:
            error_id = log_external_failure(
                logger,
                "recommendation_provider_failed",
                stage="semantic_verify",
                exc=RuntimeError(secret),
            )

        output = "\n".join(captured.output)
        self.assertNotIn(secret, output)
        self.assertNotIn("token=", output)
        self.assertIn("stage=semantic_verify", output)
        self.assertIn("error_type=RuntimeError", output)
        self.assertIn(f"error_id={error_id}", output)

    def test_error_id_is_stable_for_stage_and_exception_type(self) -> None:
        first = safe_error_id("steam_owned_games", RuntimeError("first secret"))
        second = safe_error_id("steam_owned_games", RuntimeError("second secret"))
        other = safe_error_id("semantic_verify", RuntimeError("first secret"))

        self.assertEqual(first, second)
        self.assertNotEqual(first, other)
        self.assertRegex(first, r"^[0-9a-f]{10}$")

    def test_supports_astrbot_logger_facade_without_log_method(self) -> None:
        logger = Mock(spec=["debug", "info", "warning", "error", "critical"])

        log_external_failure(
            logger,
            "provider_failed",
            stage="preference_parse",
            exc=RuntimeError("secret token=abcdef"),
        )

        output = " ".join(str(value) for value in logger.warning.call_args.args)
        self.assertNotIn("abcdef", output)
        self.assertIn("RuntimeError", output)

    def test_logs_allowlisted_code_status_and_unique_correlation_id(self) -> None:
        logger = logging.getLogger("safe-errors-correlation-test")
        error = TypedExternalError()

        with self.assertLogs(logger, level="WARNING") as captured:
            log_external_failure(
                logger,
                "steam_failed",
                stage="owned_games",
                exc=error,
            )
            log_external_failure(
                logger,
                "steam_failed",
                stage="owned_games",
                exc=error,
            )

        output = "\n".join(captured.output)
        self.assertIn("error_code=steam_http_status", output)
        self.assertIn("status_code=403", output)
        correlation_ids = re.findall(r"correlation_id=([0-9a-f]{12})", output)
        self.assertEqual(len(correlation_ids), 2)
        self.assertNotEqual(correlation_ids[0], correlation_ids[1])


class TypedExternalError(RuntimeError):
    code = "steam_http_status"
    status_code = 403


if __name__ == "__main__":
    unittest.main()
