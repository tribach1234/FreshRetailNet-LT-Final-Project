"""Dependency-free checks for the local FreshRetailNet-LT project."""
from pathlib import Path
import struct
import sys

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {'train.parquet': 7_869_549, 'eval.parquet': 70_000}


def main():
    print('=' * 64)
    print('FRESHRETAIL-LT | STEP 1: FILE CHECK')
    print('=' * 64)
    print('Project directory:', ROOT)
    print('Python:', sys.version.split()[0], '| executable:', sys.executable)
    errors = []
    for name in EXPECTED:
        path = ROOT / 'data' / 'raw' / name
        if not path.exists():
            print('[MISSING]', path)
            errors.append(name)
            continue
        size = path.stat().st_size
        with path.open('rb') as f:
            head = f.read(4)
            if size >= 8:
                f.seek(-8, 2)
                tail = f.read(8)
            else:
                tail = b''
        footer_ok = len(tail) == 8 and tail[4:] == b'PAR1'
        header_ok = head == b'PAR1'
        footer_len = struct.unpack('<I', tail[:4])[0] if footer_ok else None
        print(f'[{"OK" if header_ok and footer_ok else "INVALID"}] {name}: '
              f'{size / 1024**2:,.2f} MiB, Parquet header={header_ok}, footer={footer_ok}')
        if footer_ok:
            print(f'      Footer metadata size: {footer_len:,} bytes')
        if not (header_ok and footer_ok):
            errors.append(name)
    if errors:
        print('\nFAILED: check download and folder paths:', ', '.join(errors))
        sys.exit(1)
    print('\nPASS: Both files have the expected Parquet signatures.')


if __name__ == '__main__':
    main()
