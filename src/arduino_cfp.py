import re, sys
from collections import Counter, OrderedDict

MAPPINGS = OrderedDict([
    # Core digital/analog I/O
    (r'\bdigitalRead\s*\(', 'E'),
    (r'\banalogRead\s*\(', 'E'),
    (r'\bdigitalWrite\s*\(', 'X'),
    (r'\banalogWrite\s*\(', 'X'),
    (r'\bdelay\s*\(', 'X'),
    # Serial
    (r'\bSerial\.begin\s*\(', 'X'),
    (r'\bSerial\.read(?:String)?\s*\(', 'E'),
    (r'\bSerial\.available\s*\(', 'E'),
    (r'\bSerial\.print(?:ln)?\s*\(', 'X'),
    (r'\bSerial\.write\s*\(', 'X'),
    # SoftwareSerial
    (r'\bSoftwareSerial\.begin\s*\(', 'X'),
    (r'\bSoftwareSerial\.read(?:String)?\s*\(', 'E'),
    (r'\bSoftwareSerial\.available\s*\(', 'E'),
    (r'\bSoftwareSerial\.print(?:ln)?\s*\(', 'X'),
    (r'\bSoftwareSerial\.write\s*\(', 'X'),
    # LCD / Displays
    (r'\blcd\.begin\s*\(', 'X'),
    (r'\blcd\.(?:print|println|setCursor|clear|home)\s*\(', 'X'),
    (r'\bu8g2\.begin\s*\(', 'X'),
    (r'\bu8g2\.(?:print|printStr|draw[A-Za-z]*)\s*\(', 'X'),
    # Networking
    (r'\bWiFi\.begin\s*\(', 'X'),
    (r'\bWiFiClient\.read\s*\(', 'E'),
    (r'\bWiFiClient\.available\s*\(', 'E'),
    (r'\bWiFiClient\.write\s*\(', 'X'),
    (r'\bEthernet\.begin\s*\(', 'X'),
    (r'\bEthernetClient\.read\s*\(', 'E'),
    (r'\bEthernetClient\.available\s*\(', 'E'),
    (r'\bEthernetClient\.write\s*\(', 'X'),
    (r'\bHTTPClient\.(?:GET|POST|PUT|PATCH|DELETE)\s*\(', 'X'),
    # I²C / SPI
    (r'\bWire\.begin\s*\(', 'X'),
    (r'\bWire\.requestFrom\s*\(', 'E'),
    (r'\bWire\.read[A-Za-z]*\s*\(', 'E'),
    (r'\bWire\.write[A-Za-z]*\s*\(', 'X'),
    (r'\bSPI\.begin\s*\(', 'X'),
    (r'\bSPI\.transfer\s*\(', 'X'),
    # Servos & Actuators
    (r'\bServo\.attach\s*\(', 'X'),
    (r'\bServo\.write[A-Za-z]*\s*\(', 'X'),
    (r'\btone\s*\(', 'X'),
    (r'\bnoTone\s*\(', 'X'),
    # EEPROM
    (r'\bEEPROM\.read\s*\(', 'R'),
    (r'\bEEPROM\.get\s*\(', 'R'),
    (r'\bEEPROM\.write\s*\(', 'W'),
    (r'\bEEPROM\.put\s*\(', 'W'),
    # SD / FS / SPIFFS
    (r'\bSD\.begin\s*\(', 'X'),
    (r'\bSD\.open\s*\(', 'E'),
    (r'\bFile\.read[A-Za-z]*\s*\(', 'R'),
    (r'\bFile\.available\s*\(', 'R'),
    (r'\bFile\.write[A-Za-z]*\s*\(', 'W'),
    (r'\bFile\.print(?:ln)?\s*\(', 'W'),
    (r'\bSPIFFS\.begin\s*\(', 'X'),
    (r'\bSPIFFS\.open\s*\(', 'E'),
    (r'\bLittleFS\.begin\s*\(', 'X'),
    (r'\bLittleFS\.open\s*\(', 'E'),
    # Generic fallbacks (last for generality)
    (r'\b\w+\.read[A-Za-z]*\s*\(', 'E'),
    (r'\b\w+\.write[A-Za-z]*\s*\(', 'X'),
    (r'\b\w+\.available\s*\(', 'E'),
    (r'\b\w+\.begin\s*\(', 'X'),
    (r'\b\w+\.open\s*\(', 'E'),
    (r'\b\w+\.get[A-Za-z]*\s*\(', 'E'),
    (r'\b\w+\.set[A-Za-z]*\s*\(', 'X'),
    (r'\b\w+\.draw[A-Za-z]*\s*\(', 'X'),
])


def analyze_file(path):
    per_line = {}
    totals = Counter({'E': 0, 'X': 0, 'R': 0, 'W': 0})
    setup_pattern = r'void\s+setup\s*\(\s*\)\s*\{'
    loop_pattern = r'void\s+loop\s*\(\s*\)\s*\{'
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for lineno, line in enumerate(f, start=1):
            code = line.split('//')[0].strip()
            counts = Counter({'E': 0, 'X': 0, 'R': 0, 'W': 0})
            if re.search(setup_pattern, code) or re.search(loop_pattern, code):
                counts['E'] += 1  # Triggering Entry

            # Track used positions to avoid double-counting
            used_spans = set()
            for pattern, cat in MAPPINGS.items():
                for match in re.finditer(pattern, code):
                    start, end = match.start(), match.end()
                    # Check if this span overlaps with any used span
                    overlap = any(start < u_end and end > u_start for u_start, u_end in used_spans)
                    if not overlap:
                        counts[cat] += 1
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
    hdr = f"{'Ln':>3} │ {'E':>2} {'X':>2} {'R':>2} {'W':>2} │ CFP │ Code"
    sep = "─────┼────┼────┼────┼────┼────┼────────────────────────────────────"
    print(hdr)
    print(sep)
    for ln, info in per_line.items():
        print(f"{ln:3} │ {info['E']:2} {info['X']:2} {info['R']:2} {info['W']:2} │ {info['CFP']:3} │ {info['code']}")
    print(sep)
    print(f"{'Tot':>3} │ {totals['E']:2} {totals['X']:2} {totals['R']:2} {totals['W']:2} │ {totals['CFP']:3}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 arduino_cfp.py <your_sketch.ino>")
        sys.exit(1)
    per_line, totals = analyze_file(sys.argv[1])
    print_report(per_line, totals)