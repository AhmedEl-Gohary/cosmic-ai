import re
import csv
import sys
import os
from collections import Counter, OrderedDict

class COSMICAnalyzer:
    CSV_FILE = 'cosmic_dataset.csv'

    C_MAPPINGS = [
        (r'\b(?:int|float|double|char)\s+([^;]+);', 'var_init'),
        (r'\b(\w+)\s+\w+\s*\(([^)]*)\)\s*\{', 'func_header'),
        (r'\breturn\b\s+([^;]+);', 'return_stmt'),
        (r'\bscanf\s*\(([^)]+)\)\s*;', 'scanf'),
        (r'\bprintf\s*\(([^)]+)\)\s*;', 'W'),
        (r'\b\w+\s*=\s*[0-9]+(?:\.[0-9]+)?\s*;', 'W'),
        (r'\b\w+\s*=\s*\w+\s*;', 'RW'),
        (r'\b\w+\s*=\s*[^=]+\s*[+\-*/]\s*[^;]+;', 'RW'),
        (r'\b\w+\s*(?:\+=|-=|\+\+|--)\s*[^;]*;', 'RW'),
        (r'\b\w+\s*(?:==|!=|>|<|>=|<=)\s*[0-9]+(?:\.[0-9]+)?', 'R'),
        (r'\b\w+\s*(?:==|!=|>|<|>=|<=)\s*\w+', 'RR'),
    ]

    ARDUINO_MAPPINGS = OrderedDict([
        (r'\bdigitalRead\s*\(', 'E'), (r'\banalogRead\s*\(', 'E'),
        (r'\bdigitalWrite\s*\(', 'X'), (r'\banalogWrite\s*\(', 'X'),
        (r'\bdelay\s*\(', 'X'),
        (r'\bSerial\.begin\s*\(', 'X'),
        (r'\bSerial\.read(?:String)?\s*\(', 'E'),
        (r'\bSerial\.available\s*\(', 'E'),
        (r'\bSerial\.print(?:ln)?\s*\(', 'X'),
        (r'\bSerial\.write\s*\(', 'X'),
        (r'\bSoftwareSerial\.begin\s*\(', 'X'),
        (r'\bSoftwareSerial\.read(?:String)?\s*\(', 'E'),
        (r'\bSoftwareSerial\.available\s*\(', 'E'),
        (r'\bSoftwareSerial\.print(?:ln)?\s*\(', 'X'),
        (r'\bSoftwareSerial\.write\s*\(', 'X'),
        (r'\blcd\.begin\s*\(', 'X'),
        (r'\blcd\.(?:print|println|setCursor|clear|home)\s*\(', 'X'),
        (r'\bu8g2\.begin\s*\(', 'X'),
        (r'\bu8g2\.(?:print|printStr|draw[A-Za-z]*)\s*\(', 'X'),
        (r'\bWiFi\.begin\s*\(', 'X'),
        (r'\bWiFiClient\.read\s*\(', 'E'),
        (r'\bWiFiClient\.available\s*\(', 'E'),
        (r'\bWiFiClient\.write\s*\(', 'X'),
        (r'\bEthernet\.begin\s*\(', 'X'),
        (r'\bEthernetClient\.read\s*\(', 'E'),
        (r'\bEthernetClient\.available\s*\(', 'E'),
        (r'\bEthernetClient\.write\s*\(', 'X'),
        (r'\bHTTPClient\.(?:GET|POST|PUT|PATCH|DELETE)\s*\(', 'X'),
        (r'\bWire\.begin\s*\(', 'X'),
        (r'\bWire\.requestFrom\s*\(', 'E'),
        (r'\bWire\.read[A-Za-z]*\s*\(', 'E'),
        (r'\bWire\.write[A-Za-z]*\s*\(', 'X'),
        (r'\bSPI\.begin\s*\(', 'X'),
        (r'\bSPI\.transfer\s*\(', 'X'),
        (r'\bServo\.attach\s*\(', 'X'),
        (r'\bServo\.write[A-Za-z]*\s*\(', 'X'),
        (r'\btone\s*\(', 'X'),
        (r'\bnoTone\s*\(', 'X'),
        (r'\bEEPROM\.read\s*\(', 'R'),
        (r'\bEEPROM\.get\s*\(', 'R'),
        (r'\bEEPROM\.write\s*\(', 'W'),
        (r'\bEEPROM\.put\s*\(', 'W'),
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
        (r'\b\w+\.read[A-Za-z]*\s*\(', 'E'),
        (r'\b\w+\.write[A-Za-z]*\s*\(', 'X'),
        (r'\b\w+\.available\s*\(', 'E'),
        (r'\b\w+\.begin\s*\(', 'X'),
        (r'\b\w+\.open\s*\(', 'E'),
        (r'\b\w+\.get[A-Za-z]*\s*\(', 'E'),
        (r'\b\w+\.set[A-Za-z]*\s*\(', 'X'),
        (r'\b\w+\.draw[A-Za-z]*\s*\(', 'X'),
    ])

    def __init__(self):
        # write header if missing
        if not os.path.exists(self.CSV_FILE):
            with open(self.CSV_FILE, 'w', newline='') as f:
                csv.writer(f, quoting=csv.QUOTE_ALL).writerow(
                    ['code','E','X','R','W','CFP']
                )
            print(f"✔ Initialized new CSV: {self.CSV_FILE}")

    def analyze(self, path):
        ext = os.path.splitext(path)[1].lower()
        if ext == '.c':
            per_line, totals = self._analyze_c(path)
            print(f"✔ C-file '{os.path.basename(path)}' analyzed.")
        elif ext in ('.ino',):
            per_line, totals = self._analyze_arduino(path)
            print(f"✔ Arduino-file '{os.path.basename(path)}' analyzed.")
        else:
            raise ValueError("Unsupported extension: "+ext)

        self._append_csv(per_line)
        print(f"→ All lines added to {self.CSV_FILE} successfully!")
        self._print_report(per_line, totals)
        print("🎉 Analysis complete and data saved! 🎉")

    def _analyze_c(self, path):
        per_line = {}
        totals = Counter({'E':0,'X':0,'R':0,'W':0})
        with open(path,'r',encoding='utf-8',errors='ignore') as f:
            for ln, raw in enumerate(f,1):
                line = raw.strip()
                # skip empty, preprocessor, comment-only, braces
                if not line or line.startswith('#') or line.startswith('//') or line in ('{','}'):
                    continue
                code = line.split('//')[0].strip()
                if not code:
                    continue

                counts = Counter({'E':0,'X':0,'R':0,'W':0})
                used = set()
                for pat,cat in self.C_MAPPINGS:
                    for m in re.finditer(pat, code):
                        s,e = m.span()
                        if any(s<ue and e>us for us,ue in used): continue
                        used.add((s,e))
                        if   cat=='func_header':
                            r,p = m.group(1), m.group(2)
                            if r!='void': counts['X']+=1
                            counts['E'] += self.count_parameters(p)
                        elif cat=='return_stmt':
                            counts['R'] += self.count_reads_in_expression(m.group(1))
                        elif cat=='scanf':
                            counts['R'] += self.count_scanf_reads(m.group(1))
                        elif cat=='W':
                            counts['W'] +=1
                        elif cat=='RW':
                            counts['R'] +=1; counts['W'] +=1
                        elif cat=='RR':
                            counts['R'] +=2
                        elif cat=='R':
                            counts['R'] +=1
                        elif cat=='var_init':
                            counts['W'] += self.count_variables(m.group(1))

                cfp = sum(counts.values())
                per_line[ln] = {'code':code, **counts, 'CFP':cfp}
                totals.update(counts)

        totals['CFP'] = sum(totals[c] for c in ('E','X','R','W'))
        return per_line, totals

    def _analyze_arduino(self, path):
        per_line = {}
        totals = Counter({'E':0,'X':0,'R':0,'W':0})
        setup = re.compile(r'void\s+setup\s*\(\s*\)\s*\{')
        loop  = re.compile(r'void\s+loop\s*\(\s*\)\s*\{')

        with open(path,'r',encoding='utf-8',errors='ignore') as f:
            for ln, raw in enumerate(f,1):
                line = raw.strip()
                if not line or line.startswith('//') or line in ('{','}'):
                    continue
                code = line.split('//')[0].strip()
                if not code: continue

                counts = Counter({'E':0,'X':0,'R':0,'W':0})
                if setup.search(code) or loop.search(code):
                    counts['E'] +=1

                used = set()
                for pat,cat in self.ARDUINO_MAPPINGS.items():
                    for m in re.finditer(pat, code):
                        s,e = m.span()
                        if any(s<ue and e>us for us,ue in used): continue
                        counts[cat] +=1
                        used.add((s,e))

                cfp = sum(counts.values())
                per_line[ln] = {'code':code, **counts, 'CFP':cfp}
                totals.update(counts)

        totals['CFP'] = sum(totals[c] for c in ('E','X','R','W'))
        return per_line, totals

    def _append_csv(self, per_line):
        with open(self.CSV_FILE,'a',newline='') as f:
            writer = csv.writer(f, quoting=csv.QUOTE_ALL)
            for info in per_line.values():
                writer.writerow([
                    info['code'],
                    info['E'],
                    info['X'],
                    info['R'],
                    info['W'],
                    info['CFP']
                ])

    def _print_report(self, per_line, totals):
        print(f"\n{'Ln':>3} │ {'E':>2} {'X':>2} {'R':>2} {'W':>2} │ CFP │ Code")
        print("─────┼────┼────┼────┼────┼────┼────────────────────────────────────")
        for ln,info in per_line.items():
            print(f"{ln:3} │ {info['E']:2} {info['X']:2} {info['R']:2} {info['W']:2} │ {info['CFP']:3} │ {info['code']}")
        print("─────┼────┼────┼────┼────┼────┼────────────────────────────────────")
        print(f"{'Tot':>3} │ {totals['E']:2} {totals['X']:2} {totals['R']:2} {totals['W']:2} │ {totals['CFP']:3}\n")

    @staticmethod
    def count_parameters(params):
        if not params.strip() or params.strip()=='void': return 0
        return len([p for p in params.split(',') if p.strip()])

    @staticmethod
    def count_reads_in_expression(expr):
        ids = re.findall(r'\b[a-zA-Z_]\w*\b(?!\s*\()', expr)
        return len(ids) if re.search(r'[+\-*/]', expr) else (1 if ids else 0)

    @staticmethod
    def count_scanf_reads(args):
        fmt = args.split(',')[0].strip()
        return len(re.findall(r'\%\w', fmt))

    @staticmethod
    def count_variables(var_list):
        return len(re.findall(r'\b\w+\s*=\s*[^,]+', var_list))

if __name__=='__main__':
    if len(sys.argv)!=2:
        print("Usage: python cosmic_analyzer.py <file.c/.ino>")
        sys.exit(1)
    COSMICAnalyzer().analyze(sys.argv[1])
