from .full_metagenotype import full_metagenotype
from .pp_pi_metagenotype import pp_pi_metagenotype

NAMED_STRUCTURES = dict(
    default=full_metagenotype,
    full=full_metagenotype,
    full_metagenotype=full_metagenotype,
    pp_pi=pp_pi_metagenotype,
)
