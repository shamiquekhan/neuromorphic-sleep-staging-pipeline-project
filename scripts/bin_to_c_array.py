from __future__ import annotations

from pathlib import Path
import argparse


def to_c_array(data: bytes, var_name: str) -> str:
    lines = []
    lines.append('#include <cstdint>')
    lines.append('')
    lines.append(f'const unsigned char {var_name}[] = {{')

    chunk = 12
    for i in range(0, len(data), chunk):
        part = data[i : i + chunk]
        hexes = ', '.join(f'0x{b:02x}' for b in part)
        lines.append(f'    {hexes},')

    lines.append('};')
    lines.append(f'const unsigned int {var_name}_len = sizeof({var_name});')
    lines.append('')
    return '\n'.join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description='Convert binary file to C array source')
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--var', default='student_model_data')
    args = parser.parse_args()

    src = Path(args.input)
    dst = Path(args.output)
    data = src.read_bytes()
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(to_c_array(data, args.var), encoding='utf-8')
    print(f'wrote={dst} bytes={len(data)}')


if __name__ == '__main__':
    main()
