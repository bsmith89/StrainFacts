from sfacts.model_zoo import (
    base,
    new_pi_regularization,
    with_missing,
)

NAMED_STRUCTURES = dict(
    default=base.structure,
    new_regularization=new_pi_regularization.structure,
    missing=with_missing.structure,
)
