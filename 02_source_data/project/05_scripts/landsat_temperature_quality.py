"""Landsat 8 C2 L2 same-scene decoding for diagnostic use, not annual modeling.

Source: https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LC08_C02_T1_L2
No cloud-distance/uncertainty threshold or dynamic water exclusion is applied.
These functions neither resample nor merge dates. Geographic land eligibility
must be computed separately with the project's fixed land mask.
"""
import numpy as np

PIXEL_BITS = {'fill':0, 'dilated_cloud':1, 'cirrus':2, 'cloud':3,
              'cloud_shadow':4, 'snow':5}
OPTICAL_SATURATION_BITS = sum(1 << k for k in (0,1,2,3,4,5,6,8))


def decode_quality(temperature_dn, qa_pixel, qa_radsat):
    t=np.asarray(temperature_dn); q=np.asarray(qa_pixel); s=np.asarray(qa_radsat)
    if not (t.shape == q.shape == s.shape):
        raise ValueError('Temperature and quality arrays must have identical grids/shapes')
    for name,a in [('temperature',t),('qa_pixel',q),('qa_radsat',s)]:
        if a.dtype.kind not in 'ui' or np.any(a<0) or np.any(a>65535):
            raise ValueError(name+' must contain unsigned 16-bit range integer codes')
    flags={name:(q & (1 << bit))!=0 for name,bit in PIXEL_BITS.items()}
    flags['temperature_fill']=t==0
    flags['terrain_occlusion']=(s & (1<<11))!=0
    clear=(t>0) & ((q & 63)==0)
    valid=clear & ~flags['terrain_occlusion']
    temperature_c=np.full(t.shape,np.nan,dtype='float64')
    temperature_c[valid]=t[valid].astype('float64')*0.00341802+149.0-273.15
    return {'flags':flags,'clear_temperature':clear,'clear_unoccluded_temperature':valid,
            'temperature_c_clear_unoccluded':temperature_c,
            'water_flag_diagnostic':(q & (1<<7))!=0,
            'optical_saturation_diagnostic':(s & OPTICAL_SATURATION_BITS)!=0,
            'cloud_confidence':((q>>8)&3).astype('uint8'),
            'shadow_confidence':((q>>10)&3).astype('uint8'),
            'snow_confidence':((q>>12)&3).astype('uint8'),
            'cirrus_confidence':((q>>14)&3).astype('uint8')}


def scale_ancillary(values, *, scale, nodata):
    """Metadata-supplied fill is removed before scaling. Real zero remains zero."""
    a=np.asarray(values)
    valid=np.isfinite(a)
    if nodata is not None: valid &= a != nodata
    result=np.full(a.shape,np.nan,dtype='float64')
    result[valid]=a[valid].astype('float64')*scale
    return result
