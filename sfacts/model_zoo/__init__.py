from sfacts.model_zoo import (
    base,
    new_pi_regularization,
    with_missing,
    simple,
)

NAMED_STRUCTURES = dict(
    default=base.model,
    new_regularization=new_pi_regularization.model,
    missing=with_missing.model,
    simple=simple.model,
)
