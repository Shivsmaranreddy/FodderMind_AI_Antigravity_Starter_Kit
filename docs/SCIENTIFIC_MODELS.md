# FodderMind AI scientific modules

These standard-library-only modules are deliberately separate from Streamlit,
Gymnasium, PPO and the older simulation engine. They implement biomass and feed
accounting and nutrient-requirement estimates. They are **not locally validated
crop-yield, goat-growth, health or economic prediction models**.

## Installation and execution

Copy `models/fodder_model.py`, `models/feed_inventory.py`, `models/goat_model.py`
and their shared `models/scientific_common.py` into the starter kit's `models/`.
Keep the starter kit's existing `models/__init__.py` and `schemas.py`. The minimal
`__init__.py` in this standalone bundle exists only to run the bundle by itself.
Import the modules directly, e.g. `from models.fodder_model import FodderModel`.
Copy tests into `tests/scientific/` if installing in the starter kit. Run:

```text
python -m pytest tests/scientific -q -p no:cacheprovider
```

In this standalone folder, run `python -m pytest -q`. Only pytest is needed for
tests. The old dashboard and PPO environment do not automatically adopt these
equations; replacing their orchestration is a separate integration step. In
particular, do not continue using the older `digest_and_grow()` calculation and
present it as a result of this scientific goat model.

## Evidence and units

`Quantity(value, unit, provenance, lower=None, upper=None)` validates finite,
nonnegative values. Model fields enforce exact unit strings. No hidden percent,
grams, fresh-weight, per-hectare or herd-to-head conversions are allowed.

|Evidence label|Meaning|
|---|---|
|observed|A named measurement with an ISO date; measurement quality is not certified by the label.|
|sourced|A documented published estimate; local transfer remains uncertain.|
|uncertain|An uncalibrated assumption, including every test/demo crop, storage and feed-composition value.|
|synthetic|A sampled scenario or model-propagated amount, not a new observation.|

Bounds on a Quantity are user-defined **scenario ranges**, not confidence
intervals. `quantity.sample(random.Random(seed))` draws uniformly within explicit
bounds and labels the draw synthetic. The dynamics themselves are deterministic
and never touch a global random generator. Reconstruct parameter objects after
sampling so cross-parameter constraints are checked. No default empirical
uncertainty distribution is claimed. A sourced coefficient does not make its
derived requirement an observed fact.

## 1. Fodder biomass

|Variable/API field|Unit|Meaning|
|---|---|---|
|B / biomass_kg_dm|kg DM|Standing dry matter over the entire plot|
|K / capacity|kg DM|Asymptotic plot dry-matter capacity|
|r / growth_rate|1/day|Potential logistic growth rate|
|R / cutting_residual|kg DM|Minimum residual protected from cutting|
|lambda / standing_loss_rate|1/day|Continuous fractional standing-loss hazard|
|h / harvest_loss_fraction|fraction|Fraction of removed biomass lost during handling|
|s / stress_factor|fraction [0,1]|Caller-supplied limitation factor|
|dt / days|day|Elapsed biological time, including fractional days|

The exact logistic transition for constant r and s over dt is:

`B_after_growth = K*B / (B + (K-B)*exp(-r*s*dt))`.

This is the analytical solution of `dB/dt = r*s*B*(1-B/K)`. It represents a
bounded phenomenological regrowth curve; it is not asserted to be a published
Napier calibration. B=0 stays zero: establishment must be provided as initial
biomass. s=0 gives no growth, but separately applied field losses may continue.

`standing_loss = B*(1-exp(-lambda*dt))`; then subtract the loss.

`harvest_removed = min(requested, max(B-R,0))`;
`harvest_loss = h*harvest_removed`; `usable_harvest = (1-h)*harvest_removed`.

Each event satisfies `B_after = B_before + growth - standing_loss - harvest_removed`.
Handling loss belongs to removed biomass and is not subtracted from the field
twice. Only `usable_harvest` enters inventory. `grow`, `apply_standing_losses`,
and `harvest` are separate calls; the orchestrator selects and records their
order. Do not call each once per UI refresh. The R bound prevents overcutting;
it is not an artificial floor against mortality. Field losses may reduce B below
R or to zero. Longer integration steps with changing stress or interleaved
harvest are not equivalent to a single constant-stress step.

All five crop parameters are required, with evidence metadata; there are no
production defaults. No rainfall-to-growth conversion, soil-water balance,
fertility response, hydroponic fresh-weight multiplier, disease or seasonal
senescence curve is invented. Sampling r alone does not represent all uncertainty.

## 2. Feed inventory and composition

Inventory state is kg DM in receipt-ordered batches. Each batch carries its own
immutable feed composition, receipt evidence, loss-rate evidence and shelf-life
evidence. A shared feed name does not overwrite different batch analyses.

`DM_kg = as_fed_kg * DM_fraction`; fractions are 0-1, not percent. A DM fraction
of 0.20 therefore maps 10 kg fresh feed to 2 kg DM. Missing DM content prevents
as-fed conversion. The inverse is only an equivalent at the specified moisture
content; water uptake/evaporation is not simulated.

Available composition fields are ME (MJ/kg DM), CP, NDF, calcium and phosphorus
(each kg nutrient/kg DM). They are optional; missing is **not zero**. ME must
be 0-45 MJ/kg DM (broad sanity bound, not a recommended feed range). Fractions
must be 0-1. CP and NDF can overlap analytically and must not be forced to sum
with other fields to one. No universal feed library values are embedded.

For nutrient j, `amount_j = sum(batch_DM_i * concentration_ij)`. `nutrient(j)`
returns a known subtotal, the DM of feeds missing that analysis, evidence, and
`total=None` whenever any positive-DM portion is missing j. Thus an incomplete
energy analysis cannot produce a false assurance of energy sufficiency.

At `advance_to(t)`, apply first-order DM decay only until each batch's expiry:
`decay = batch_DM * (1-exp(-loss_rate * elapsed_days))`. At the exact expiry
time, discard the residual as `expired_kg_dm`; do not also count it as decay.
`None` shelf life explicitly means no modeled expiry, not evidence of safe
indefinite storage. Already expired receipts are rejected. Older receipts need
their actual `age_days`; the received mass is measured on arrival, so earlier
losses are not applied twice. Time cannot move backwards.

Advance the inventory clock **before** withdrawals at a new time. A withdrawal
returns only available FIFO mass, optionally filtered by feed name. It does
not imply the goats ate that amount; refusals require separate accounting.
No automatic stock appears after a field harvest. The ledger identity is:

`received = stock + withdrawn + decay + expired`.

Stored DM composition is held constant while mass decays. This nonselective
loss assumption is a placeholder, unsuitable for predicting fermentation,
mycotoxins or selective protein/energy deterioration. CP is not converted into
metabolizable protein. Simulated stock/withdrawal quantities are labelled
synthetic even if their original receipt was measured.

## 3. Goat requirements, not growth forecasts

Animal profiles record individual live weight (kg BW), age (days), sex, actual
breed/cross, physiological stage, source biotype, head count and optional body
condition (1-5). Pen results multiply individual requirements by head count;
use separate profiles for heterogeneous animals because `(mean BW)^0.75` is
not the average of individual metabolic weights.

Luo et al. [1,2] provide the implemented population equations. For post-weaning
growing animals, with W in kg and a stated target G in g/head/day:

`ME_maintenance = a*W^0.75`; `ME_target_gain = b*G/1000` (MJ/head/day).

|Source biotype|a, MJ/kg BW^0.75/day|b, MJ/kg target gain|c, g MP/g target gain|
|---|---:|---:|---:|
|Indigenous|0.489|19.8|0.290|
|Dairy|0.580|23.1|0.290|
|Meat, >=50% Boer ancestry|0.489|23.1|0.404|

`MP_maintenance = 3.07*W^0.75/1000`;
`MP_target_gain = c*G/1000` (kg MP/head/day).

The growing-MP database covered post-weaning animals up to approximately 18
months; the implementation rejects ages above 548 days for this equation.
The day threshold is an operational approximation, not a universal maturation
age. No preweaning formula is silently substituted.

For mature indigenous/dairy maintenance, [1] gives
`ME = 0.462*W^0.75` MJ/head/day and a non-browse MP formulation:
`MP = (0.0267*1000*D + 1.031*W^0.75 + 0.2*W^0.6)/1000` kg/head/day,
where D is explicitly supplied kg DM/head/day. Adult target gain is unsupported.

The paper classifications are not interchangeable with farm labels: raising a
local goat for meat does not establish Boer ancestry. Pregnancy, lactation,
Angora, preweaning, substantial browse, and mature Boer defaults are explicitly
unsupported here. Selecting an appropriate physiological stage is a caller/expert
responsibility. Age, sex and condition are recorded without invented correction
coefficients. The optional maintenance-energy multiplier requires a Quantity
and evidence. Omission uses unadjusted maintenance with a warning; it does not
claim activity/heat neutrality in real Telangana conditions.

`assess_consumed` accepts an actually consumed daily herd ration. It compares
known ME with requirements. Dietary CP supply is reported separately. MP supply
must come from a separately documented diet/digestion calculation; missing MP
produces `None` for protein adequacy. No generic CP-to-MP efficiency is assumed.
Even known energy and MP sufficiency do not establish mineral, vitamin, fiber,
water, health or intake-capacity adequacy. A zero ration has zero dietary nutrient
supply. Results always leave `predicted_gain_g_head_day=None` and body weight
unchanged. Growth targets are planning inputs, not predictions or achieved gain.

## Local calibration and review required

|Needed in Telangana|Why it matters / suggested reviewer|
|---|---|
|Cultivar, plot area, cut age/height, replicated DM yield across seasons|Fit K, r, cutting residual and losses; PJTAU/local agronomist or KVK.|
|Irrigation delivery, soil moisture, temperature, rainfall and fertility records|Define stress outside this model; weekly/manual records can start calibration.|
|Batch moisture, CP, NDF, minerals, digestibility and appropriate ME estimates|Avoid treating a literature mean as a local lab result; accredited feed laboratory.|
|Storage duration, shade, ambient conditions, weighed discards and mold/toxin assessment|Fit loss rates and shelf life; unsafe feed cannot be identified by DM balance.|
|Breed/cross verification, individual BW/age, sex, condition, activity, reproductive state|Select applicable nutrient equations; PVNRTVU/veterinarian/ruminant nutritionist review.|
|Browse/tannin content, rumen degradability, microbial protein and intake/refusals|Estimate MP supply and assess limits of the non-browse assumptions.|
|Repeated independent weights, health and parasite records, feed trials|Any future growth model needs independent calibration and validation; feed mass alone is insufficient.|

Institutions above are proposed review routes, not endorsements or claimed
partnerships. No actual Telangana field dataset was supplied. Test values and
the demo ration are synthetic examples, not a feeding recommendation.

## References

1. Luo et al. (2004). *Nutrient requirements of goats: developed equations, other
   considerations and future research to improve them*. Small Ruminant Research
   53:191-219. https://doi.org/10.1016/j.smallrumres.2004.04.001
2. Luo et al. (2004). *Metabolizable protein requirements for maintenance and gain
   of growing goats*. Small Ruminant Research 53:309-326.
   https://doi.org/10.1016/j.smallrumres.2004.04.003
3. NRC (2007). *Nutrient Requirements of Small Ruminants*.
   https://doi.org/10.17226/11654 — broader expert reference; this implementation
   is not the complete NRC system.
4. Feedipedia. Elephant grass (Cenchrus purpureus).
   https://feedipedia.org/node/395 — composition varies with maturity and growing
   conditions; no site average was inserted as a Telangana parameter.
