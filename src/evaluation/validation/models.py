from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, cast

import pandas as pd

if TYPE_CHECKING:
    from src.evaluation.validation.config import ValidationThresholds


STATUS_PASS = "PASS"
STATUS_WARNING = "WARNING"
STATUS_FAIL = "FAIL"
STATUS_SKIPPED = "SKIPPED"

VALIDATION_STATUSES = (
    STATUS_PASS,
    STATUS_WARNING,
    STATUS_FAIL,
    STATUS_SKIPPED,
)


class CheckKind(StrEnum):
    INTEGRITY = "integrity"
    DIAGNOSTIC = "diagnostic"


TECHNICAL_STATUS_PASS = "PASS"
TECHNICAL_STATUS_FAIL = "FAIL"

REPORT_SCHEMA_VERSION = 2


def _missing_values_as_none(value: object) -> object:
    if isinstance(value, dict):
        return {key: _missing_values_as_none(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_missing_values_as_none(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_missing_values_as_none(item) for item in value)
    if pd.api.types.is_scalar(value) and bool(pd.isna(value)):
        return None
    return value


@dataclass(frozen=True)
class ValidationCheckResult:
    check_name: str
    status: str
    message: str
    category: str
    algorithm_name: str | None = None
    agent_name: str | None = None
    opponent_name: str | None = None
    training_episode: int | None = None
    observed_value: float | None = None
    threshold: float | None = None
    sample_size: int | None = None
    standard_error: float | None = None
    ci_lower: float | None = None
    ci_upper: float | None = None
    details: dict[str, object] | None = None
    check_type: CheckKind | None = None
    check_id: str | None = None
    hypothesis_id: str | None = None

    def __post_init__(self) -> None:
        if self.status not in VALIDATION_STATUSES:
            raise ValueError(f"Unsupported validation status: {self.status!r}")
        if self.check_type is None:
            inferred = (
                CheckKind.INTEGRITY
                if self.status == STATUS_FAIL
                else CheckKind.DIAGNOSTIC
            )
            object.__setattr__(self, "check_type", inferred)
        if self.check_type == CheckKind.DIAGNOSTIC and self.status == STATUS_FAIL:
            raise ValueError("Diagnostic checks cannot use FAIL; use WARNING.")

    @property
    def blocking(self) -> bool:
        return self.check_type == CheckKind.INTEGRITY and self.status == STATUS_FAIL

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        assert self.check_type is not None
        payload["check_type"] = self.check_type.value
        payload["check_id"] = self.check_id or self.check_name
        payload["blocking"] = self.blocking
        return cast(
            dict[str, object],
            _missing_values_as_none(payload),
        )


@dataclass(frozen=True)
class ValidationReport:
    input_path: str
    thresholds: ValidationThresholds
    checks: list[ValidationCheckResult]
    validation_mode: str = "training-opponent"
    training_episode: int | None = None
    model_selection: str | None = None
    schema_version: int = REPORT_SCHEMA_VERSION

    @property
    def technically_valid(self) -> bool:
        return not any(check.blocking for check in self.checks)

    @property
    def passed(self) -> bool:
        """Backward-compatible alias for technical validity."""

        return self.technically_valid

    @property
    def technical_status(self) -> str:
        return (
            TECHNICAL_STATUS_PASS
            if self.technically_valid
            else TECHNICAL_STATUS_FAIL
        )

    def status_counts(self) -> dict[str, int]:
        return {
            status: sum(check.status == status for check in self.checks)
            for status in VALIDATION_STATUSES
        }

    def check_type_counts(self, check_type: CheckKind) -> dict[str, int]:
        return {
            status: sum(
                check.check_type == check_type and check.status == status
                for check in self.checks
            )
            for status in VALIDATION_STATUSES
        }

    def to_dict(self) -> dict[str, object]:
        return cast(
            dict[str, object],
            _missing_values_as_none(
                {
                    "schema_version": self.schema_version,
                    "input_path": self.input_path,
                    "validation_mode": self.validation_mode,
                    "training_episode": self.training_episode,
                    "model_selection": self.model_selection,
                    "technical_status": self.technical_status,
                    "technically_valid": self.technically_valid,
                    "passed": self.passed,
                    "status_counts": self.status_counts(),
                    "integrity_check_counts": self.check_type_counts(
                        CheckKind.INTEGRITY
                    ),
                    "diagnostic_check_counts": self.check_type_counts(
                        CheckKind.DIAGNOSTIC
                    ),
                    "diagnostic_warning_count": sum(
                        check.check_type == CheckKind.DIAGNOSTIC
                        and check.status == STATUS_WARNING
                        for check in self.checks
                    ),
                    "diagnostic_warning_counts": sum(
                        check.check_type == CheckKind.DIAGNOSTIC
                        and check.status == STATUS_WARNING
                        for check in self.checks
                    ),
                    "skipped_check_count": sum(
                        check.status == STATUS_SKIPPED for check in self.checks
                    ),
                    "skipped_check_counts": sum(
                        check.status == STATUS_SKIPPED for check in self.checks
                    ),
                    "thresholds": asdict(self.thresholds),
                    "checks": [check.to_dict() for check in self.checks],
                }
            ),
        )
