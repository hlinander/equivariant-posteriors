import healpix
def interpolate_dh_to_hp(nside, variable: xr.DataArray):
    """
    Input is xr.DataArray 
    """
    
    npix = healpix.nside2npix(nside)
    hlong, hlat = healpix.pix2ang(nside, np.arange(0, npix, 1), lonlat=True, nest=True)
    hlong = np.mod(hlong, 360)
    xlong = xr.DataArray(hlong, dims="z")
    xlat = xr.DataArray(hlat, dims="z")

    xhp = variable.interp(lat=xlat, lon=xlong, kwargs={"fill_value": None})
    hp_image = np.array(xhp.to_numpy(), dtype=np.float32) # ! removed to_array()
    return hp_image

def interpolate_dh_to_hp_output(nside, variable: xr.DataArray):
    """
    Input is xr.DataArray 
    """
    
    npix = healpix.nside2npix(nside)
    hlong, hlat = healpix.pix2ang(nside, np.arange(0, npix, 1), lonlat=True, nest=True)
    hlong = np.mod(hlong, 360)
    xlong = xr.DataArray(hlong, dims="z")
    xlat = xr.DataArray(hlat, dims="z")

    xhp = variable.interp(y=xlat, x=xlong, kwargs={"fill_value": None})
    hp_image = np.array(xhp.to_numpy(), dtype=np.float32) # ! removed to_array()
    return hp_image

def e5_to_numpy_hp(e5xr, nside: int, normalized: bool):
    """
    Input is class with xr.DataArray class variables
    """

    hp_surface = interpolate_dh_to_hp(nside, e5xr.surface)
    hp_upper = interpolate_dh_to_hp(nside, e5xr.upper)

    if normalized:
        stats = deserialize_dataset_statistics(nside)
        hp_surface, hp_upper = normalize_sample(stats.item(), hp_surface, hp_upper)

    return hp_surface, hp_upper