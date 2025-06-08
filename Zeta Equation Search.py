# This script is a symbolic regression tool designed to search for symbolic equations for the 
# relating s to zeta s the Riemann zeta function either for the real or imaginary part using PySR  
# It leverages GPU based indexing for further speed and efficiency.    
# Designed for large scale interactive and fault tolerant equation discovery  
# Includes a full logging pipeline  

# Be sure to include the flat_index_gpu.py file in the same directory. And download all the library files required to run this.

import os
import re
import csv
import logging
from collections import Counter

import numpy as np
import pandas as pd
from pysr import PySRRegressor
from numba import njit, prange
from tqdm import tqdm

import tkinter as tk
from tkinter import filedialog, messagebox

from flat_index_gpu import FlatIndexGPU

# PySR symbolic regression hyperparameters making it easy for user to adjust and test with the Zeta csv file:

PYSR_BINARY_OPERATORS = ["+", "-", "*", "/", "pow"]  # Operators allowed between variables. Add/remove to allow/disallow operations (e.g., remove "pow" to avoid exponentials).
PYSR_UNARY_OPERATORS = ["sin", "cos", "exp", "log", "sqrt", "abs"]  # Functions allowed on single variables. Add/remove to control function complexity (e.g., remove "exp" to avoid exponentials).
PYSR_POPULATIONS = 1000  # Number of populations in the evolutionary search. Increase for more diversity, decrease for faster runs.
PYSR_POPULATION_SIZE = 100  # Number of individuals per population. Higher values allow more candidate equations per generation, but use more memory.
PYSR_NITERATIONS = 200  # Number of evolutionary iterations. Increase for more thorough search (slower), decrease for faster but less thorough search.
PYSR_MAXSIZE = 15  # Maximum size (complexity) of equations. Increase to allow more complex equations, decrease for simpler results.
PYSR_ELEMENTWISE_LOSS = "(x, y) -> (x - y)^2"  # Loss function for regression. Change to other loss functions for different error metrics.
PYSR_VERBOSITY = 1  # Level of output detail. 0 = silent, higher = more logging.
PYSR_CONSTRAINTS = {"pow": (-1, 1)}  # Constraints on operators. E.g., restrict "pow" exponents to between -1 and 1.
PYSR_BATCHING = True  # Whether to use batching for large datasets. Set False for small datasets.
PYSR_BATCH_SIZE = 50000  # Batch size for training. Increase for faster training if memory allows, decrease if running out of memory.
PYSR_NCYCLES_PER_ITERATION = 100  # Number of cycles per iteration. Higher = more search per iteration (slower but more thorough).

ACCURACY_THRESHOLD = 0.001  # or 1e-3, Threshold for considering an equation "accurate enough" (e.g., for filtering or stopping criteria)

# Set outputs directory inside the Riemann folder
OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)

LOG_FILENAME = os.path.join(OUTPUTS_DIR, "zeta_equation_search.log")

logging.basicConfig(
    level=logging.INFO,  # Set logging level to INFO (show info, warning, error, critical)
    format="%(asctime)s - %(levelname)s - %(message)s",  # Log message format: timestamp, level, message
    handlers=[
        logging.FileHandler(LOG_FILENAME),  # Log to file
        logging.StreamHandler()             # Also log to console
    ]
)
# This configures logging to output messages both to a log file and the console, with timestamps and log levels.

logging.info("Starting Zeta Equation Search script.")
root = tk.Tk()
root.withdraw()
mode = messagebox.askquestion("Equation Search", "Start a new search?\n(Click 'No' to continue from previous log)")

if mode == 'yes':
    logging.info("User selected 'New Search' mode.")
    csv_file_path = filedialog.askopenfilename(
        title="Select CSV File with Zeta Zeros",
        filetypes=[("CSV Files", "*.csv")]
    )
    if not csv_file_path:
        logging.error("No CSV file selected. Exiting.")
        print("No file selected. Exiting.")
        exit()
    # Save log file in outputs directory by default
    default_log = os.path.join(OUTPUTS_DIR, "equation_discovery_log.csv")
    log_file = filedialog.asksaveasfilename(
        title="Save Log File As",
        defaultextension=".csv",
        initialfile=os.path.basename(default_log),
        initialdir=OUTPUTS_DIR,
        filetypes=[("CSV Files", "*.csv")]
    )
    if not log_file:
        logging.error("No log file selected. Exiting.")
        print("No log file selected. Exiting.")
        exit()
    resume = False
    logged_equations = set()
else:
    logging.info("User selected 'Continue from Previous Log' mode.")
    log_file = filedialog.askopenfilename(
        title="Select Existing Log File",
        initialdir=OUTPUTS_DIR,
        filetypes=[("CSV Files", "*.csv")]
    )
    if not log_file:
        logging.error("No log file selected. Exiting.")
        print("No log file selected. Exiting.")
        exit()
    csv_file_path = filedialog.askopenfilename(
        title="Select CSV File with Zeta Zeros",
        filetypes=[("CSV Files", "*.csv")]
    )
    if not csv_file_path:
        logging.error("No CSV file selected. Exiting.")
        print("No file selected. Exiting.")
        exit()
    logged_equations = set()
    with open(log_file, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            logged_equations.add(row["Equation"])
    resume = True
logging.info(f"Loading CSV file: {csv_file_path}")
data = pd.read_csv(csv_file_path, delimiter=None)

def parse_s(s):
    """
    Parse a string representing a complex number (with 'i' or 'j') and return a Python complex object.
    """
    s = s.replace(' ', '').replace('i', 'j')
    if s.startswith('(') and s.endswith(')'):
        s = s[1:-1]
    return complex(s)

data['s_complex'] = data['s'].apply(parse_s)
data['s_real'] = data['s_complex'].apply(lambda x: x.real)
data['s_imag'] = data['s_complex'].apply(lambda x: x.imag)
data['zeta_real'] = data['Zeta Real'].astype(float)
data['zeta_imag'] = data['Zeta Imag'].astype(float)
X = data[['s_real', 's_imag']].values
y_real = data['zeta_real'].values
y_imag = data['zeta_imag'].values

part = messagebox.askquestion("Regression Target", "Fit real part of zeta(s)?\n(Click 'No' for imaginary part)")
if part == 'yes':
    y_vals = y_real
    logging.info("Fitting real part of zeta(s).")
else:
    y_vals = y_imag
    logging.info("Fitting imaginary part of zeta(s).")

@njit(parallel=True, fastmath=True)
def evaluate_equation(func, x_values):
    """
    Evaluate a given function 'func' on each row of x_values in parallel using Numba for speed.
    Returns an array of predictions.
    """
    y_pred = np.zeros(len(x_values))
    for i in prange(len(x_values)):
        y_pred[i] = func(x_values[i])
    return y_pred

model = PySRRegressor(
    binary_operators=PYSR_BINARY_OPERATORS,
    unary_operators=PYSR_UNARY_OPERATORS,
    populations=PYSR_POPULATIONS,
    population_size=PYSR_POPULATION_SIZE,
    niterations=PYSR_NITERATIONS,
    maxsize=PYSR_MAXSIZE,
    elementwise_loss=PYSR_ELEMENTWISE_LOSS,
    verbosity=PYSR_VERBOSITY,
    constraints=PYSR_CONSTRAINTS,
    batching=PYSR_BATCHING,
    batch_size=PYSR_BATCH_SIZE,
    ncycles_per_iteration=PYSR_NCYCLES_PER_ITERATION
)

while True:
    try:
        logging.info("Starting symbolic regression with PySR.")
        model.fit(X, y_vals)
        logging.info("Symbolic regression completed.")
        def extract_equation_features(eqn_str):
            """
            Extract features from an equation string:
            Counts of each operator/function, depth (parentheses), and total size.
            Returns a list of feature values.
            """
            ops = ["+", "-", "*", "/", "pow", "sin", "cos", "exp", "log", "sqrt"]
            counts = Counter()
            for op in ops:
                if op in ["+", "-", "*", "/"]:
                    counts[op] = eqn_str.count(op)
                else:
                    counts[op] = len(re.findall(rf"\b{op}\b", eqn_str))
            depth = eqn_str.count("(")
            size = sum(counts.values()) + 1
            return [counts[op] for op in ops] + [depth, size]
        write_header = not (resume and os.path.exists(log_file))
        all_eqn_features = []
        all_eqn_strings = []
        with open(log_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            op_headers = ["Add", "Sub", "Mul", "Div", "Pow", "Sin", "Cos", "Exp", "Log", "Sqrt", "Depth", "Size"]
            if file.tell() == 0:
                writer.writerow(
                    ["Equation"] + op_headers + ["MSE"]  
                )
            try:
                logging.info("Evaluating equations generated by PySR.")
                for _, eqn in tqdm(model.equations_.iterrows(), desc="Evaluating equations", unit="eqn"):
                    eqn_str = eqn['equation']
                    if eqn_str in logged_equations:
                        continue
                    try:
                        eqn_func = eqn['lambda_format']
                        logging.debug(f"Equation: {eqn_str}, lambda_format: {eqn_func}, type: {type(eqn_func)}")
                    except Exception as e:
                        logging.error(f"Failed to get lambda for equation {eqn_str}: {e}", exc_info=True)
                        print(f"Failed to get lambda for equation {eqn_str}: {e}")
                        eqn_features = extract_equation_features(eqn_str)
                        writer.writerow(
                            [eqn_str] + eqn_features + ["FAILED"]  
                        )
                        continue
                    try:
                        y_preds = eqn_func(X)
                    except Exception as eval_e:
                        logging.error(f"Failed to evaluate equation {eqn_str} with x_vals.reshape(-1, 1): {eval_e}", exc_info=True)
                        print(f"Failed to evaluate equation {eqn_str} with x_vals.reshape(-1, 1): {eval_e}")
                        eqn_features = extract_equation_features(eqn_str)
                        writer.writerow(
                            [eqn_str] + eqn_features + ["FAILED"]  
                        )
                        continue
                    eqn_features = extract_equation_features(eqn_str)
                    try:
                        mse = np.mean((y_preds - y_vals) ** 2)
                        y_mean = np.mean(y_vals)
                        ss_tot = np.sum((y_vals - y_mean) ** 2)
                        ss_res = np.sum((y_vals - y_preds) ** 2)
                        score = 1 - ss_res / ss_tot if ss_tot != 0 else float('nan')
                    except Exception as mse_e:
                        logging.error(f"Failed to compute MSE/Score for {eqn_str}: {mse_e}")
                        writer.writerow(
                            [eqn_str] + eqn_features + ["FAILED"]  
                        )
                        continue
                    all_eqn_features.append(eqn_features)
                    all_eqn_strings.append(eqn_str)

                    writer.writerow(
                        [eqn_str] + eqn_features + [f"{mse:.5f}"]  
                    )
                    print(f"Equation: {eqn_str}, MSE: {mse:.5f}, Score: {score:.5f}") 
                    logging.info(f"Logged Equation: {eqn_str}, MSE: {mse:.5f}, Score: {score:.5f}")
            except KeyboardInterrupt:
                logging.warning("Search interrupted by user. Progress saved.")
                print("Search interrupted by user. All processed equations have been saved.")
                break
        if all_eqn_features:
            logging.info("Building GPU index for equations.")
            features_np = np.array(all_eqn_features, dtype=np.float32)
            index = FlatIndexGPU(dim=features_np.shape[1], dtype='float32')
            index.add(features_np)
            logging.info("GPU index built successfully.")
            ids, dists = index.search(features_np[:1], topk=3)
            print("Nearest equations to the first equation (by feature vector):")
            for rank, (idx, dist) in enumerate(zip(ids[0], dists[0]), 1):
                print(f"Rank {rank}: Equation: {all_eqn_strings[idx]}, Distance: {dist:.4f}")
        logging.info(f"All equations logged in {log_file}")
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        messagebox.showerror("Error", f"Unexpected error: {e}\nCheck log file for details.")
        break
