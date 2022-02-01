from sfacts.model_zoo import (
    base,
    base_ssdd,
    with_missing,
    simple,
    simple_ssdd,
    simple_no_rho,
    simple2,
    simple2_with_error,
    simple_ssdd2,
    simple_ssdd2_with_error,
    simple_like_sfinder,
    ssdd3,
    ssdd3_with_error,
    ssdd3_with_error_with_missing,
    ssdd4_with_error,
    simplest_simulation,
    ssdd3_with_error_lowmem,
)

NAMED_STRUCTURES = dict(
    default=base.model,
    ssdd=base_ssdd.model,
    missing=with_missing.model,
    simple=simple2.model,
    simple_with_error=simple2_with_error.model,
    simple_ssdd=simple_ssdd.model,
    simple_no_rho=simple_no_rho.model,
    simple_ssdd2=simple_ssdd2.model,
    simple_ssdd2_with_error=simple_ssdd2_with_error.model,
    simple_like_sfinder=simple_like_sfinder.model,
    ssdd3=ssdd3.model,
    ssdd3_with_error=ssdd3_with_error.model,
    ssdd3_with_error_with_missing=ssdd3_with_error_with_missing.model,
    ssdd4_with_error=ssdd4_with_error.model,
    simplest_simulation=simplest_simulation.model,
    ssdd3_with_error_lowmem=ssdd3_with_error_lowmem.model,
)
