"""FIFO dry-matter inventory; no fodder growth or animal-growth inference.

kg DM = kg as-fed * DM_fraction. Nutrient amount = sum(kg DM * concentration).
ME concentration uses MJ/kg DM; CP, NDF, Ca and P use kg/kg DM (fractions).
CP and NDF are analytical measures with potential overlap: do not sum them as
disjoint components. MP supply is not computed from CP with a fixed multiplier.
Unknown composition remains unknown; known subtotal is not a complete total.

For each batch, storage DM loss is B*(1-exp(-lambda*dt)) until expiry. Residual
DM at expiry is discarded separately. Composition is assumed constant in retained
DM, an uncertain nonselective-loss approximation. Call advance_to(day) explicitly
before decisions at a new time. FIFO applies to receipt order, not expiry order.
"""
from dataclasses import dataclass
import math
from .scientific_common import Quantity, Provenance, Evidence, number


NUTRIENTS = {
    'me': ('me_mj_per_kg_dm', 'MJ'),
    'cp': ('cp_fraction_dm', 'kg CP'),
    'ndf': ('ndf_fraction_dm', 'kg NDF'),
    'calcium': ('calcium_fraction_dm', 'kg Ca'),
    'phosphorus': ('phosphorus_fraction_dm', 'kg P'),
}


@dataclass(frozen=True)
class FeedComposition:
    name: str
    dm_fraction_as_fed: Quantity | None = None
    me_mj_per_kg_dm: Quantity | None = None
    cp_fraction_dm: Quantity | None = None
    ndf_fraction_dm: Quantity | None = None
    calcium_fraction_dm: Quantity | None = None
    phosphorus_fraction_dm: Quantity | None = None

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError('feed name required')
        if self.dm_fraction_as_fed is not None:
            self.dm_fraction_as_fed.checked('fraction', positive=True, maximum=1)
        if self.me_mj_per_kg_dm is not None:
            self.me_mj_per_kg_dm.checked('MJ/kg DM', maximum=45)
        for field, _ in list(NUTRIENTS.values())[1:]:
            value = getattr(self, field)
            if value is not None:
                value.checked('fraction', maximum=1)

    def as_fed_to_dm(self, kg_as_fed):
        mass = number(kg_as_fed, 'kg_as_fed')
        if self.dm_fraction_as_fed is None:
            raise ValueError('dry-matter fraction required for as-fed conversion')
        return mass*self.dm_fraction_as_fed.value


@dataclass(frozen=True)
class FeedPortion:
    composition: FeedComposition
    kg_dm: float
    quantity_provenance: Provenance

    def __post_init__(self):
        number(self.kg_dm, 'kg_dm')
        if not isinstance(self.composition, FeedComposition) or not isinstance(self.quantity_provenance, Provenance):
            raise ValueError('composition and quantity provenance required')

    @property
    def kg_as_fed_equivalent(self):
        """Equivalent at supplied DM fraction, not a moisture-dynamics prediction."""
        dm = self.composition.dm_fraction_as_fed
        return None if dm is None else number(self.kg_dm/dm.value,'as-fed equivalent')


@dataclass(frozen=True)
class NutrientAmount:
    known_subtotal: float
    unknown_feed_kg_dm: float
    unit: str
    evidence: tuple[Provenance, ...]

    def __post_init__(self):
        number(self.known_subtotal,'nutrient subtotal')
        number(self.unknown_feed_kg_dm,'unanalysed DM')

    @property
    def total(self):
        return self.known_subtotal if self.unknown_feed_kg_dm == 0 else None


@dataclass(frozen=True)
class Ration:
    """A physical collection of feed portions; not necessarily eaten by an animal."""
    portions: tuple[FeedPortion, ...]

    def __post_init__(self):
        if not isinstance(self.portions, tuple) or not all(isinstance(p, FeedPortion) for p in self.portions):
            raise ValueError('portions must be an immutable tuple of FeedPortion')
        number(self.kg_dm, 'total ration DM')

    @property
    def kg_dm(self):
        return math.fsum(p.kg_dm for p in self.portions)

    def nutrient(self, name):
        if name not in NUTRIENTS:
            raise ValueError(f'unknown nutrient {name!r}')
        field, unit = NUTRIENTS[name]
        known, unknown, evidence = [], [], []
        for portion in self.portions:
            if portion.kg_dm == 0:
                continue
            value = getattr(portion.composition, field)
            if value is None:
                unknown.append(portion.kg_dm)
            else:
                known.append(portion.kg_dm*value.value)
                evidence.extend((value.provenance, portion.quantity_provenance))
        return NutrientAmount(math.fsum(known), math.fsum(unknown), unit, tuple(evidence))


@dataclass
class _Batch:
    composition: FeedComposition
    kg_dm: float
    quantity_provenance: Provenance
    decay_rate: Quantity
    expires_day: float | None
    shelf_life: Quantity | None


@dataclass(frozen=True)
class StorageLoss:
    decay_kg_dm: float
    expired_kg_dm: float

    @property
    def total_kg_dm(self):
        return self.decay_kg_dm+self.expired_kg_dm


class FeedInventory:
    def __init__(self, start_day=0.):
        self._day = number(start_day, 'start_day')
        self._batches = []
        self._received = self._withdrawn = self._decayed = self._expired = 0.

    @property
    def day(self):
        return self._day

    @property
    def stock_kg_dm(self):
        return math.fsum(b.kg_dm for b in self._batches)

    @property
    def balance(self):
        return dict(received_kg_dm=self._received, withdrawn_kg_dm=self._withdrawn,
                    decay_kg_dm=self._decayed, expired_kg_dm=self._expired,
                    stock_kg_dm=self.stock_kg_dm,
                    error_kg_dm=self.stock_kg_dm-(self._received-self._withdrawn-self._decayed-self._expired))

    def snapshot(self):
        return Ration(tuple(FeedPortion(b.composition,b.kg_dm,self._derived_evidence(b)) for b in self._batches))

    @staticmethod
    def _derived_evidence(batch):
        return Provenance(Evidence.SYNTHETIC, 'FIFO inventory mass-balance calculation',
            f'Receipt source: {batch.quantity_provenance.source}; receipt status: '
            f'{batch.quantity_provenance.status.value}. Modelled mass is not a fresh measurement.')

    def storage_assumptions(self):
        """Immutable quantities retain their source/placeholder tags for audit."""
        return tuple((b.composition.name,b.decay_rate,b.shelf_life) for b in self._batches)

    def receive(self, composition, amount: Quantity, decay_rate: Quantity,
                shelf_life_days: Quantity | None, age_days=0.):
        """Receive kg DM or kg as-fed; storage age already elapsed is explicit.
        Amount is the mass actually arriving, so earlier losses are not reapplied.
        shelf_life_days=None explicitly means no modeled expiry, not proven safety.
        """
        if not isinstance(composition, FeedComposition):
            raise ValueError('FeedComposition required')
        rate = decay_rate.checked('1/day')
        age = number(age_days, 'age_days')
        if amount.unit == 'kg DM':
            dm = amount.checked('kg DM')
        elif amount.unit == 'kg as-fed':
            dm = composition.as_fed_to_dm(amount.checked('kg as-fed'))
        else:
            raise ValueError('quantity must use kg DM or kg as-fed')
        expires = None
        if shelf_life_days is not None:
            life = shelf_life_days.checked('day', positive=True)
            if age >= life:
                raise ValueError('cannot receive expired feed')
            expires = number(self.day+life-age, 'expiry time')
        number(self._received+dm, 'cumulative receipts')
        if dm:
            self._batches.append(_Batch(composition,dm,amount.provenance,decay_rate,expires,shelf_life_days))
            self._received += dm
        return dm

    def advance_to(self, day):
        end = number(day, 'day', minimum=self.day)
        decayed = expired = 0.
        remaining = []
        for batch in self._batches:
            stop = min(end,batch.expires_day) if batch.expires_day is not None else end
            dt = max(0.,stop-self.day)
            loss = batch.kg_dm*(-math.expm1(-batch.decay_rate.value*dt))
            batch.kg_dm = max(0.,batch.kg_dm-loss)
            decayed += loss
            if batch.expires_day is not None and end >= batch.expires_day:
                expired += batch.kg_dm
            elif batch.kg_dm:
                remaining.append(batch)
        self._batches = remaining
        self._day = end
        self._decayed += decayed
        self._expired += expired
        return StorageLoss(decayed, expired)

    def withdraw(self, requested_kg_dm, feed_name=None):
        """Returns actual FIFO removal; shortages appear as smaller returned mass."""
        requested = number(requested_kg_dm, 'requested_kg_dm')
        if feed_name is not None and (not isinstance(feed_name,str) or not feed_name.strip()):
            raise ValueError('feed_name must be a nonempty string or None')
        portions, outstanding = [], requested
        for batch in self._batches:
            if feed_name is not None and batch.composition.name != feed_name:
                continue
            take = min(outstanding,batch.kg_dm)
            if take:
                portions.append(FeedPortion(batch.composition,take,self._derived_evidence(batch)))
                batch.kg_dm -= take
                outstanding = max(0.,outstanding-take)
        self._batches = [b for b in self._batches if b.kg_dm > 0]
        result = Ration(tuple(portions))
        self._withdrawn += result.kg_dm
        return result
