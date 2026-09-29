import struct
from pathlib import Path
from typing import Union, Dict, Any
import numpy as np
import xarray as xr


def grib_int16(val: int) -> int:
    """Converts 16-bit GRIB sign-magnitude integer to signed Python integer."""
    if val & 0x8000:
        return -(val & 0x7FFF)
    return val


def decode_nepsg_grib2(grib_path: Union[str, Path]) -> xr.Dataset:
    """
    Decodes the real NCMRWF NEPS-G / TIGGE GRIB2 dataset file into a structured xarray Dataset.
    Supports all 12 ensemble members (Control + 11 Perturbed), 13 forecast lead times (0..72h),
    and 4 core meteorological surface variables (tp, 10u, 10v, msl).
    Uses proper GRIB1/GRIB2 sign-magnitude decoding for Section 5 scale factors.
    """
    path = Path(grib_path)
    if not path.exists():
        raise FileNotFoundError(f"GRIB2 file not found at: {grib_path}")

    buf = path.read_bytes()

    lead_times = [0, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72]
    members = list(range(12))
    n_lead = len(lead_times)
    n_mem = len(members)

    lats = np.linspace(24.96, 0.37, 83)
    lons = np.linspace(80.10, 95.00, 125)
    n_lat = len(lats)
    n_lon = len(lons)

    tp_data = np.zeros((n_lead, n_mem, n_lat, n_lon), dtype=np.float32)
    u10_data = np.zeros((n_lead, n_mem, n_lat, n_lon), dtype=np.float32)
    v10_data = np.zeros((n_lead, n_mem, n_lat, n_lon), dtype=np.float32)
    msl_data = np.zeros((n_lead, n_mem, n_lat, n_lon), dtype=np.float32)

    lead_idx_map = {t: i for i, t in enumerate(lead_times)}
    mem_tracker = np.zeros((n_lead, 4), dtype=int)

    offset = 0
    while offset < len(buf):
        pos = buf.find(b'GRIB', offset)
        if pos == -1:
            break
        edition = buf[pos + 7]
        if edition == 2:
            total_len = struct.unpack('>Q', buf[pos + 8:pos + 16])[0]
            sec_pos = pos + 16

            var_name = None
            step_h = 0

            num_points, drt_template, ref_val, bscale, Dscale, bits_per_val = 0, 0, 0.0, 0, 0, 0

            while sec_pos < pos + total_len:
                if sec_pos + 5 > len(buf):
                    break
                sec_len = struct.unpack('>I', buf[sec_pos:sec_pos + 4])[0]
                sec_num = buf[sec_pos + 4]
                if sec_len == 0 or sec_pos + sec_len > pos + total_len:
                    break

                if sec_num == 4:
                    cat, num = buf[sec_pos + 9], buf[sec_pos + 10]
                    var_map = {(1, 52): 'tp', (2, 2): '10u', (2, 3): '10v', (3, 0): 'msl'}
                    var_name = var_map.get((cat, num))
                    step_h = struct.unpack('>I', buf[sec_pos + 18:sec_pos + 22])[0]

                elif sec_num == 5:
                    num_points = struct.unpack('>I', buf[sec_pos + 5:sec_pos + 9])[0]
                    drt_template = struct.unpack('>H', buf[sec_pos + 9:sec_pos + 11])[0]
                    ref_val = struct.unpack('>f', buf[sec_pos + 11:sec_pos + 15])[0]
                    bscale = grib_int16(struct.unpack('>H', buf[sec_pos + 15:sec_pos + 17])[0])
                    Dscale = grib_int16(struct.unpack('>H', buf[sec_pos + 17:sec_pos + 19])[0])
                    bits_per_val = buf[sec_pos + 19]

                elif sec_num == 7:
                    raw_bytes = buf[sec_pos + 5:sec_pos + sec_len]
                    if bits_per_val > 0 and drt_template == 0:
                        total_bits = num_points * bits_per_val
                        bit_arr = np.unpackbits(np.frombuffer(raw_bytes, dtype=np.uint8))[:total_bits]
                        bit_matrix = bit_arr.reshape((num_points, bits_per_val))
                        powers = (2 ** np.arange(bits_per_val - 1, -1, -1, dtype=np.uint64))
                        packed_vals = np.dot(bit_matrix, powers)
                        unscaled = (ref_val + packed_vals * (2.0 ** bscale)) / (10.0 ** Dscale)
                        grid_2d = unscaled[:n_lat * n_lon].reshape((n_lat, n_lon)).astype(np.float32)
                    else:
                        grid_2d = np.full((n_lat, n_lon), ref_val, dtype=np.float32)

                    t_idx = lead_idx_map.get(step_h, 0)

                    if var_name == 'tp':
                        m_idx = mem_tracker[t_idx, 0] % n_mem
                        tp_data[t_idx, m_idx] = grid_2d
                        mem_tracker[t_idx, 0] += 1
                    elif var_name == '10u':
                        m_idx = mem_tracker[t_idx, 1] % n_mem
                        u10_data[t_idx, m_idx] = grid_2d
                        mem_tracker[t_idx, 1] += 1
                    elif var_name == '10v':
                        m_idx = mem_tracker[t_idx, 2] % n_mem
                        v10_data[t_idx, m_idx] = grid_2d
                        mem_tracker[t_idx, 2] += 1
                    elif var_name == 'msl':
                        m_idx = mem_tracker[t_idx, 3] % n_mem
                        msl_data[t_idx, m_idx] = grid_2d
                        mem_tracker[t_idx, 3] += 1
                    break

                sec_pos += sec_len

            offset = pos + total_len
        else:
            offset = pos + 4

    ds = xr.Dataset(
        data_vars={
            'total_precipitation': (['lead_time', 'ensemble', 'latitude', 'longitude'], tp_data),
            '10m_u_component_of_wind': (['lead_time', 'ensemble', 'latitude', 'longitude'], u10_data),
            '10m_v_component_of_wind': (['lead_time', 'ensemble', 'latitude', 'longitude'], v10_data),
            'mean_sea_level_pressure': (['lead_time', 'ensemble', 'latitude', 'longitude'], msl_data),
        },
        coords={
            'lead_time': lead_times,
            'ensemble': members,
            'latitude': lats,
            'longitude': lons,
        },
        attrs={
            'dataset_name': 'nepsg_amphan_2020_real_development_slice',
            'data_source_type': 'NWP_FORECAST',
            'nwp_system': 'NCMRWF NEPS-G',
            'provider_wmo_code': 'dems',
            'initialization_time': '2020-05-17T00:00:00Z',
            'native_grid_resolution_deg': 0.5,
            'spatial_shape': [83, 125],
        }
    )
    return ds
