from sfacts.model_zoo import (
    base,
    new_pi_regularization,
    with_missing,
    simple,
    simple_with_error,
    simple_dirichlet,
)

NAMED_STRUCTURES = dict(
    default=base.model,
    new_regularization=new_pi_regularization.model,
    missing=with_missing.model,
    simple=simple.model,
    simple_dirichlet=simple_dirichlet.model,
    simple_error=simple_with_error.model,
)
