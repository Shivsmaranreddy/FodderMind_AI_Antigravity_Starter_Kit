"""Strict quantities and evidence metadata shared by scientific modules (stdlib only)."""
from dataclasses import dataclass, replace
from datetime import date
from enum import Enum
import math
from numbers import Real
import random


def number(value, name, minimum=0., maximum=None, positive=False):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f'{name} must be a finite real number')
    if value < minimum or (positive and value <= minimum) or (maximum is not None and value > maximum):
        raise ValueError(f'{name} is outside its permitted range')
    return float(value)


def integer(value, name, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return value


class Evidence(str, Enum):
    OBSERVED = 'observed'
    SOURCED = 'sourced'
    UNCERTAIN = 'uncertain'
    SYNTHETIC = 'synthetic'


@dataclass(frozen=True)
class Provenance:
    status: Evidence
    source: str
    note: str = ''
    measured_on: str | None = None

    def __post_init__(self):
        if not isinstance(self.status, Evidence):
            raise ValueError('status must be an Evidence enum')
        if not isinstance(self.source, str) or not self.source.strip():
            raise ValueError('source or placeholder rationale is required')
        if self.status == Evidence.OBSERVED and not self.measured_on:
            raise ValueError('observed values require an ISO measurement date')
        if self.measured_on is not None:
            date.fromisoformat(self.measured_on)


PLACEHOLDER = Provenance(Evidence.UNCERTAIN, 'User-selected research placeholder',
                         'Requires local Telangana measurements and expert review.')


@dataclass(frozen=True)
class Quantity:
    """Nonnegative value, exact unit string and evidence. Bounds are scenario ranges,
    not confidence intervals. Sampling requires an explicit local random generator.
    """
    value: float
    unit: str
    provenance: Provenance
    lower: float | None = None
    upper: float | None = None

    def __post_init__(self):
        number(self.value, 'value')
        if not isinstance(self.unit, str) or not self.unit.strip():
            raise ValueError('unit required')
        if not isinstance(self.provenance, Provenance):
            raise ValueError('provenance required')
        if (self.lower is None) != (self.upper is None):
            raise ValueError('provide both lower and upper bounds')
        if self.lower is not None:
            number(self.lower, 'lower')
            number(self.upper, 'upper')
            if not self.lower <= self.value <= self.upper:
                raise ValueError('bounds must enclose value')

    def checked(self, unit, *, minimum=0., maximum=None, positive=False):
        if self.unit != unit:
            raise ValueError(f'expected {unit!r}, got {self.unit!r}; convert explicitly')
        value = number(self.value, unit, minimum, maximum, positive)
        for bound in (self.lower, self.upper):
            if bound is not None:
                number(bound, 'scenario bound', minimum, maximum, positive)
        return value

    def sample(self, rng: random.Random):
        if not isinstance(rng, random.Random):
            raise ValueError('supply random.Random(seed) explicitly')
        if self.lower is None:
            raise ValueError('sampling needs explicit scenario bounds')
        return replace(self, value=rng.uniform(self.lower, self.upper),
                       provenance=Provenance(Evidence.SYNTHETIC, self.provenance.source,
                       'Uniform scenario draw, not a fitted probability distribution'))
