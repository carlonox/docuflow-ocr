"""Deterministic field validation.

OCR output is probabilistic; validation is the deterministic gate that turns
raw text into trustworthy data. Validators are small composable rules:
regex, length, allowed choices, checksums. A document extractor combines them
into a validation plan and the cascade's final stage applies it.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field as dfield
from typing import Any, Dict, List, Optional, Pattern, Sequence


@dataclass
class ValidationResult:
    """Outcome of validating a single field.

    Attributes:
        field: Field name that was validated.
        valid: Whether the value passed all rules.
        value: The (possibly normalized) value.
        errors: Human-readable reasons for failure.
        normalized: Normalized form of the value (e.g. digits only).
    """

    field: str
    valid: bool = False
    value: Optional[str] = None
    errors: List[str] = dfield(default_factory=list)
    normalized: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for logging and learning hooks."""
        return {
            "valid": self.valid,
            "value": self.value,
            "errors": self.errors,
            "normalized": self.normalized,
        }


class Validator(ABC):
    """Base class for field validators."""

    @abstractmethod
    def validate(self, field: str, value: Optional[str]) -> ValidationResult:
        """Validate one field value."""
        raise NotImplementedError


class RegexValidator(Validator):
    """Validate a value against a regular expression.

    Args:
        pattern: Regex the value must match (fully, via ``fullmatch``).
        description: Optional human-readable rule description.
    """

    def __init__(self, pattern: str, description: str = "") -> None:
        self.pattern: Pattern[str] = re.compile(pattern)
        self.description = description or pattern

    def validate(self, field: str, value: Optional[str]) -> ValidationResult:
        if value is None:
            return ValidationResult(field=field, value=None, errors=["missing value"])
        match = self.pattern.fullmatch(value.strip())
        return ValidationResult(
            field=field,
            valid=match is not None,
            value=value.strip(),
            errors=[] if match else [f"does not match {self.description}"],
        )


class LengthValidator(Validator):
    """Validate a value's length (after stripping non-digits, optional)."""

    def __init__(self, min_length: int, max_length: int, digits_only: bool = False) -> None:
        self.min_length = min_length
        self.max_length = max_length
        self.digits_only = digits_only

    def validate(self, field: str, value: Optional[str]) -> ValidationResult:
        if value is None:
            return ValidationResult(field=field, value=None, errors=["missing value"])
        normalized = re.sub(r"\D", "", value) if self.digits_only else value.strip()
        length = len(normalized)
        ok = self.min_length <= length <= self.max_length
        return ValidationResult(
            field=field,
            valid=ok,
            value=value.strip(),
            normalized=normalized if self.digits_only else None,
            errors=[] if ok else [f"length {length} outside [{self.min_length}, {self.max_length}]"],
        )


class ChoiceValidator(Validator):
    """Validate that a value is one of an allowed set of choices."""

    def __init__(self, choices: Sequence[str], case_insensitive: bool = False) -> None:
        self.choices = [c.lower() for c in choices] if case_insensitive else list(choices)
        self.case_insensitive = case_insensitive

    def validate(self, field: str, value: Optional[str]) -> ValidationResult:
        if value is None:
            return ValidationResult(field=field, value=None, errors=["missing value"])
        candidate = value.strip().lower() if self.case_insensitive else value.strip()
        ok = candidate in self.choices
        return ValidationResult(
            field=field,
            valid=ok,
            value=value.strip(),
            errors=[] if ok else [f"'{value}' not in allowed choices"],
        )


class ChecksumValidator(Validator):
    """Validate a value against a checksum function.

    Args:
        checksum_fn: Callable ``(digits: str) -> bool``.
        description: Rule description used in error messages.
    """

    def __init__(self, checksum_fn, description: str = "checksum") -> None:
        self.checksum_fn = checksum_fn
        self.description = description

    def validate(self, field: str, value: Optional[str]) -> ValidationResult:
        if value is None:
            return ValidationResult(field=field, value=None, errors=["missing value"])
        digits = re.sub(r"\D", "", value)
        ok = bool(digits) and bool(self.checksum_fn(digits))
        return ValidationResult(
            field=field,
            valid=ok,
            value=value.strip(),
            normalized=digits,
            errors=[] if ok else [f"failed {self.description}"],
        )


class CompositeValidator(Validator):
    """Run multiple validators; all must pass.

    Args:
        rules: Ordered validators to apply.
        require_all: When True (default) every rule must pass; when False,
            at least one must pass.
    """

    def __init__(self, rules: Sequence[Validator], require_all: bool = True) -> None:
        self.rules = list(rules)
        self.require_all = require_all

    def validate(self, field: str, value: Optional[str]) -> ValidationResult:
        results = [rule.validate(field, value) for rule in self.rules]
        passed = [r for r in results if r.valid]
        ok = (len(passed) == len(results)) if self.require_all else bool(passed)
        errors = [error for r in results for error in r.errors if not r.valid]
        return ValidationResult(
            field=field,
            valid=ok,
            value=value.strip() if value else None,
            normalized=next((r.normalized for r in results if r.normalized), None),
            errors=errors,
        )
