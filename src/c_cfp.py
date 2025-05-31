import re
from collections import Counter

# Mapping of C instructions to COSMIC data movements with regular expressions
MAPPINGS = [
    (r'\b(?:int|float|double|char)\s+([^;]+);', 'var_init'),  # Variable initialization
    (r'\b(\w+)\s+\w+\s*\(([^)]*)\)\s*\{', 'func_header'),  # Function header
    (r'\breturn\b\s+([^;]+);', 'return_stmt'),  # Return statement
    (r'\bscanf\s*\(([^)]+)\)\s*;', 'scanf'),  # Scanf call
    (r'\bprintf\s*\(([^)]+)\)\s*;', 'W'),  # Printf call
    (r'\b\w+\s*=\s*[0-9]+(?:\.[0-9]+)?\s*;', 'W'),  # Assignment with constant
    (r'\b\w+\s*=\s*\w+\s*;', 'RW'),  # Assignment with variable
    (r'\b\w+\s*=\s*[^=]+\s*[+\-*/]\s*[^;]+;', 'RW'),  # Arithmetic instruction (1R, 1W)
    (r'\b\w+\s*(?:\+=|-=|\+\+|--)\s*[^;]*;', 'RW'),  # Increment
    (r'\b\w+\s*(?:==|!=|>|<|>=|<=)\s*[0-9]+(?:\.[0-9]+)?', 'R'),  # Logical expression type 1
    (r'\b\w+\s*(?:==|!=|>|<|>=|<=)\s*\w+', 'RR'),  # Logical expression type 2
]


def count_parameters(params):
    """Count the number of parameters in a function header."""
    if not params.strip() or params.strip() == 'void':
        return 0
    return len([p for p in params.split(',') if p.strip()])


def count_reads_in_expression(expression):
    """Count the number of variables read in an expression."""
    # Count variables in arithmetic or standalone return
    identifiers = re.findall(r'\b[a-zA-Z_]\w*\b(?!\s*\()', expression)
    return len(identifiers) if re.search(r'[+\-*/]', expression) else 1 if identifiers else 0


def count_scanf_reads(args):
    """Count the number of format specifiers in scanf."""
    format_str = args.split(',')[0].strip()
    return len(re.findall(r'\%\w', format_str))


def count_variables(var_list):
    """Count the number of variable initializations in the variable list."""
    pattern = r'\b\w+\s*=\s*[^,]+'
    matches = re.findall(pattern, var_list)
    return len(matches)

def analyze_file(path):
    """Analyze C source file and count COSMIC data movements."""
    per_line = {}
    totals = Counter({'E': 0, 'X': 0, 'R': 0, 'W': 0})

    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()

    for lineno, line in enumerate(lines, start=1):
        code = line.split('//')[0].strip()
        # Skip preprocessor directives
        if code.startswith('#'):
            per_line[lineno] = {'code': line.rstrip('\n'), 'E': 0, 'X': 0, 'R': 0, 'W': 0, 'CFP': 0}
            continue

        counts = Counter({'E': 0, 'X': 0, 'R': 0, 'W': 0})

        # Track used positions to avoid double-counting
        used_spans = set()
        for pattern, cat in MAPPINGS:
            for match in re.finditer(pattern, code):
                start, end = match.start(), match.end()
                overlap = any(start < u_end and end > u_start for u_start, u_end in used_spans)
                if not overlap:
                    if cat == 'func_header':
                        return_type = match.group(1)
                        params = match.group(2)
                        if return_type != 'void':
                            counts['X'] += 1
                        counts['E'] += count_parameters(params)
                    elif cat == 'return_stmt':
                        expression = match.group(1)
                        counts['R'] += count_reads_in_expression(expression)
                    elif cat == 'scanf':
                        args = match.group(1)
                        counts['R'] += count_scanf_reads(args)
                    elif cat == 'W':
                        counts['W'] += 1
                    elif cat == 'RW':
                        counts['R'] += 1
                        counts['W'] += 1
                    elif cat == 'RR':
                        counts['R'] += 2
                    elif cat == 'R':
                        counts['R'] += 1
                    elif cat == 'var_init':
                        var_list = match.group(1)
                        print(f"Line {lineno}: var_list = '{var_list}")
                        counts['W'] += count_variables(var_list)
                    used_spans.add((start, end))

        cfp = sum(counts[cat] for cat in ('E', 'X', 'R', 'W'))
        per_line[lineno] = {
            'code': line.rstrip('\n'),
            'E': counts['E'],
            'X': counts['X'],
            'R': counts['R'],
            'W': counts['W'],
            'CFP': cfp
        }
        totals.update(counts)

    totals['CFP'] = sum(totals[cat] for cat in ('E', 'X', 'R', 'W'))
    return per_line, totals


def print_report(per_line, totals):
    """Print a detailed report of COSMIC data movements."""
    hdr = f"{'Ln':>3} │ {'E':>2} {'X':>2} {'R':>2} {'W':>2} │ CFP │ Code"
    sep = "─────┼────┼────┼────┼────┼────┼────────────────────────────────────"
    print(hdr)
    print(sep)
    for ln, info in per_line.items():
        print(f"{ln:3} │ {info['E']:2} {info['X']:2} {info['R']:2} {info['W']:2} │ {info['CFP']:3} │ {info['code']}")
    print(sep)
    print(f"{'Tot':>3} │ {totals['E']:2} {totals['X']:2} {totals['R']:2} {totals['W']:2} │ {totals['CFP']:3}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python3 cosmic_cfp.py <your_program.c>")
        sys.exit(1)
    per_line, totals = analyze_file(sys.argv[1])
    print_report(per_line, totals)