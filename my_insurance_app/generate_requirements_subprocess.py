#!/usr/bin/env python3
import subprocess
import sys
from pathlib import Path

def generate_requirements(output_file='requirements1.txt'):
    """
    Run `pip freeze` in this environment and write the output to requirements1.txt.
    """
    # Ensure we're writing relative to the project root
    out_path = Path(__file__).parent / output_file
    with out_path.open('w') as f:
        subprocess.run([sys.executable, '-m', 'pip', 'freeze'], stdout=f, check=True)
    print(f"✔ Requirements written to {out_path}")

if __name__ == '__main__':
    generate_requirements()
