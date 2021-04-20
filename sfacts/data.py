
from sfacts.logging_util import info
from sfacts.pandas_util import idxwhere
import xarray as xr


def load_input_data(allpaths):
    data = []
    for path in allpaths:
        info(path)
        d = xr.open_dataarray(path).squeeze()
        info(f"Shape: {d.sizes}.")
        data.append(d)
    info("Concatenating data from {} files.".format(len(data)))
    data = xr.concat(data, "library_id", fill_value=0)
    info(f"Finished concatenating data: {data.sizes}")
    return data

def select_informative_positions(data, incid_thresh):
    minor_allele_incid = (data > 0).mean("library_id").min("allele")
    informative_positions = idxwhere(
        minor_allele_incid.to_series() > incid_thresh
    )
    return informative_positions
