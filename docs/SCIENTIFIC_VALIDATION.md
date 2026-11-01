# Scientific-module validation

Implemented and tested in the requested sequence:

1. Fodder module: 11 tests passed before inventory implementation.
2. Feed inventory: 11 module tests passed before goat implementation.
3. Goat model: 19 module tests passed.
4. Integrated review: 46 tests passed in 0.38 seconds after adding end-to-end
   conservation, evidence tracking and additional numerical edge cases.
5. Installed the four module files, tests and documentation in the original
   Desktop starter kit. All 46 scientific tests passed there in 3.25 seconds.
   Existing simulation, dashboard, schemas and package exports were not changed.

Coverage includes analytical logistic transitions, time partitioning under
constant conditions, extinction, carrying capacity, residual cutting limits,
standing/handling loss separation, FIFO composition preservation, expiry and
arrival age, DM/as-fed conversions, missing nutrient analyses, g/kg conversions,
herd scaling, published coefficient arithmetic, unsupported animal states,
non-finite/negative/bool inputs, no invented growth, seeded replay and a 30-day
field-to-inventory-to-consumption conservation check.

Runtime used: Python 3.12.10, pytest 9.2.0, Windows. Scientific modules themselves
use only the Python standard library. Tests use pytest. No external service or
sensor is required at run time.

Passing tests establish implementation consistency, not real-world calibration
or predictive accuracy. No local farm dataset or independent validation was
available. No crop yield, goat growth or savings claim is made. See
SCIENTIFIC_MODELS.md for unsupported states and local-data requirements.
