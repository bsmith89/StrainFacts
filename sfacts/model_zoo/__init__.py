from sfacts.model_zoo import (
    base,
    base_ssdd,
    with_missing,
    simple,
    simple_ssdd,
    simple_no_rho,
    simple2,
    simple_ssdd2,
    simple_ssdd2_with_error,
)

NAMED_STRUCTURES = dict(
    default=base.model,
    ssdd=base_ssdd.model,
    missing=with_missing.model,
    simple=simple2.model,
    simple_ssdd=simple_ssdd.model,
    simple_no_rho=simple_no_rho.model,
    simple_ssdd2=simple_ssdd2.model,
    simple_ssdd2_with_error=simple_ssdd2_with_error.model,
)
