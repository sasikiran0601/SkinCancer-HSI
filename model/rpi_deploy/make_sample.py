"""
Creates a valid test_sample.npy file for RPi inference testing.
Shape: (116, 50, 50) float32 — 116 spectral bands, 50x50 spatial pixels
Run: python make_sample.py
"""
import struct
import random
import os

def make_npy(path, shape=(116, 50, 50)):
    dtype_str = '<f4'  # little-endian float32
    header_str = "{'descr': '%s', 'fortran_order': False, 'shape': %s, }" % (dtype_str, str(shape))
    header_bytes = header_str.encode('latin1')
    # Align the complete NumPy header and place the newline last.
    padding = (64 - (10 + 2 + len(header_bytes) + 1) % 64) % 64
    header_bytes = header_bytes + b' ' * padding + b'\n'

    magic   = b'\x93NUMPY'
    version = b'\x01\x00'
    hdr_len = struct.pack('<H', len(header_bytes))

    n = 1
    for s in shape:
        n *= s

    # Generate random float32 values in [0, 1]
    random.seed(42)
    floats = [random.random() for _ in range(n)]
    data_bytes = struct.pack('<%df' % n, *floats)

    with open(path, 'wb') as f:
        f.write(magic + version + hdr_len + header_bytes + data_bytes)

    size_kb = os.path.getsize(path) / 1024
    print("Created : %s" % path)
    print("Shape   : %s  dtype: float32" % str(shape))
    print("Size    : %.1f KB" % size_kb)
    print("Ready to copy to Raspberry Pi!")

if __name__ == '__main__':
    out = os.path.join(os.path.dirname(__file__), 'test_sample.npy')
    make_npy(out)
