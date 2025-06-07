# Using the mpmath library for computing the Riemann zeta function is preferred over a custom implementation for 
# several important reasons:

# Numerical Stability and Accuracy:
# The Riemann zeta function, especially for large imaginary parts (large |Im(s)|), is numerically challenging. mpmath.zeta uses advanced algorithms (like the Riemann-Siegel formula) that are much more stable and accurate than simple series expansions.

# Performance:
# The Dirichlet eta series (alternating series) converges very slowly for large |Im(s)|, making it impractical for high values of t. mpmath automatically chooses the most efficient method for the given input, providing results much faster.

# Edge Cases and Special Values:
# mpmath.zeta handles special cases

# Precision Control:
# mpmath allows you to set arbitrary precision, which is essential for research or high-accuracy computations.

# Reliability:
# The library is well-tested and widely used in the mathematical and scientific Python community, reducing the risk of subtle bugs.

import mpmath
import csv
import os
import tkinter as tk
from tkinter import filedialog

# Try to import tqdm for progress bar, otherwise set tqdm to None
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

# Use mpmath's built-in zeta function, which uses the Riemann-Siegel formula for large |Im(s)| on the critical line.
# This is much more efficient and numerically stable for large t than the Dirichlet eta series.
# See: http://mpmath.org/doc/current/functions/zeta.html

def main():
    # Use Tkinter to open a folder selection dialog for saving the CSV file
    root = tk.Tk()
    root.withdraw()
    folder = filedialog.askdirectory(title="Select Folder to Save CSV")
    if not folder:
        print("No folder selected.")
        return
    file_path = os.path.join(folder, "zeta_values.csv")

    # Prepare 500 values of s on the critical line (Re(s) = 0.5), starting at t=1000
    # s = 0.5 + it, with t from 1000 to 1049.9 in steps of 0.1
    s_values = [0.5 + (1000 + i * 0.1) * 1j for i in range(500)]

    # Open CSV file for writing
    with open(file_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['s', 'Zeta Real', 'Zeta Imag'])  # Write CSV header

        # Use tqdm for progress bar if available, otherwise use plain iterator
        iterator = s_values
        if tqdm:
            iterator = tqdm(s_values, desc="Computing zeta values")
        else:
            print("tqdm not installed, progress bar will not be shown.")

        # Compute zeta(s) for each s and write to CSV
        for s in iterator:
            z = mpmath.zeta(s)  # Efficient computation using Riemann-Siegel for large t
            writer.writerow([
                str(s).replace('(', '').replace(')', ''),  # Format s as string, remove parentheses
                float(z.real),  # Real part of zeta(s)
                float(z.imag)   # Imaginary part of zeta(s)
            ])

    print(f"CSV saved to: {file_path}")

if __name__ == "__main__":
    main()
