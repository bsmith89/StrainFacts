from sfacts.model_zoo import simplest_simulation, ssdd3_with_error, new_model_1

NAMED_STRUCTURES = dict(
    default=ssdd3_with_error.model,
    default_simulation=simplest_simulation.model,
    ssdd3_with_error=ssdd3_with_error.model,
    simplest_simulation=simplest_simulation.model,
    new_model_1=new_model_1.model,
)
