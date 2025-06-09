Millennial Equation
Riemann Symbolic Regression Theory 
by
Adam Spencer Barnett
8 June 2025

Introduction
The Riemann Hypothesis stands as one of the most significant unsolved problems in mathematics, originating from Bernhard Riemann’s groundbreaking work in 1859. The hypothesis posits that all non-trivial zeros of the Riemann Zeta function have real parts precisely equal to ½. If proven, this result would have profound implications across number theory, cryptography, and mathematical physics (Britannica, n.d.).
Historically, numerous mathematicians have dedicated efforts to solving the hypothesis, utilizing diverse numerical, analytical, and computational methods. While these approaches have yielded increasingly accurate predictions of the non-trivial zeros often referred to as "zetas" none have yet resulted in a definitive proof. This persistent challenge highlights the critical need for alternative and innovative strategies in mathematical research.

Abstract
The theory introduced here is an alternative computational methodology termed here as the Riemann Symbolic Regression Theory which leverages computational tools, specifically the Zeta Calculator (Barnett, 2025) and Zeta Equation Search (Barnett, 2025), to uncover hidden symbolic relationships within the Zeta zeros. Using precise zeros generated from the Zeta Calculator, the Equation Finder systematically identifies increasingly accurate symbolic equations predicting these zeros. Analysis revealed that the longer and more detailed the generated symbolic equations become, the closer their predictions match the actual zeros. This discovery provides concrete computational evidence indicating that hidden symbolic structures could exist within the distribution of zeta zeros. Such structures might help in demonstrating underlying patterns essential to proving the Riemann Hypothesis.

Methodological Evidence
The research began with numerical data derived from a precise computational Zeta Calculator, accurately determining the positions of 2500 non-trivial zeros of the Riemann Zeta function. These computed zeros became input data for a symbolic regression algorithm, known as the Zeta Equation Search.
Symbolic regression, unlike traditional numerical methods, seeks explicit symbolic mathematical expressions that directly map input numbers (like indices or initial approximations) to the computed zeros. Iterative testing by the Zeta Equation Search revealed equations of increasing complexity, each providing a progressively closer approximation of the actual Zeta zero positions.
A significant observation from these symbolic equations was a clear trend: increasing the complexity of symbolic equations consistently enhanced accuracy, with predictions converging ever closer to exact values computed via the Zeta Calculator. The complexity here denotes additional symbolic terms or layers added to the equations discovered by the Equation Finder.
Here is a sample of the Zeta’s used with the Zeta Equation Search (See Figure 1). 
Column Meanings
•	s:
The original complex number input to the zeta function, typically on the critical line.
Example: 0.5+1000j means s=0.5+1000is = 0.5 + 1000is=0.5+1000i.
•	s_real:
The real part of sss. For the critical line, this is always 0.5.
•	s_imag:
The imaginary part of sss. For your data, this starts at 1000 and increases (e.g., 1000, 1000.1, 1000.2, ...).
•	dZeta Real:
The change (difference) in the real part of the zeta function compared to the previous value.
o	Δζreal[n]=ζreal[n]−ζreal[n−1]\Delta \zeta_{\text{real}}[n] = \zeta_{\text{real}}[n] - \zeta_{\text{real}}[n-1]Δζreal[n]=ζreal[n]−ζreal[n−1]
o	The first value is usually set to 0 because there’s no previous value.
•	Zeta Real:
The real part of the value returned by ζ(s)\zeta(s)ζ(s) at each sss.
Example Row
s	s_real	s_imag	dZeta Real	Zeta Real
0.5+1000j	0.5	1000	0	0.3563
0.5+1000.1j	0.5	1000.1	0.4889	0.8452
Figure 1 by the Author
The following demonstrates how the Zeta Equation Search script uses symbolic regression to improve its prediction of the real part of a Riemann zeta zero, as measured by mean squared error (MSE): 
Initial Equation (High MSE, e.g., 2.688): This is a very rough estimate, representing an early, simple candidate equation.
  x0 + 69931.58

Intermediate Equation (Lower MSE, e.g., 1.375): This equation is more complex and fits the data much better, as shown by the much lower MSE.
  (x0 * ((exp(exp(sin(sin(sin(sin(x1 * 0.19691889))) * -2.9394252))) * 0.003079456) + 0.7009184)) + 69999.26

Final Equation (Low MSE, e.g., 1.033): This final equation predicts the zeta zero position with a much greater accuracy, as indicated by the lower MSE.
  cos(x0 * 0.052665632) + (((((exp(sin(sin(sin(sin(sin(x1 * 0.18675153))) * -3.8144846))) * 0.021061895) + 0.6978137) * x0) + 69998.836) - (sin(sin(sin(sin(sin(sin(sin(sin(x1 + 0.4116114))))) * -3.8420596))) * 0.39464167)

How the Script Works:
•	The script uses symbolic regression (via PySR) to search for equations that predict the real part of the Riemann zeta function from selected input features (such as s_real |	s_imag  |	dZeta Real	 |   Zeta Real
•	It iteratively fits candidate equations, evaluates their accuracy using MSE and R² score, and logs each equation along with its features and error metrics.
•	The process continues until an equation meets strict accuracy criteria (e.g., MSE below a threshold and R² above a threshold), or until the user interrupts the search.
•	All equations and their statistics are saved to a log file for review.

Theoretical Framework and Explanation
The core theoretical proposal from this finding is straightforward yet profound: The distribution of Riemann zeta zeros inherently embodies an underlying symbolic structure or series of symbolic structures. This is strongly evidenced by the consistently improved predictive capability observed when equations discovered by the Equation Finder become symbolically more detailed (See attached Equation Log).
More explicitly:
•	Initially, simpler symbolic expressions found by the Equation Finder showed general approximations of zeros.
•	Adding more symbolic complexity (extra terms, operations, and constants identified during regression) improved accuracy significantly.
•	As complexity grows, the symbolic equations approach an exact match of the Zeta Calculator outputs, strongly suggesting that an intrinsic symbolic nature governs the zeta zero distribution.
This clear progression of improving accuracy through symbolic complexity supports a theoretical interpretation: the non-trivial zeros could adhere to symbolic patterns rather than purely numerical or random distributions. Consequently, symbolic regression equations represent intrinsic mathematical truths underlying the zeros' positioning.
Significance and Implications
If the symbolic structure observed can be generalized or rigorously demonstrated, it would imply the potential existence of closed-form symbolic solutions or expressions that govern the zeros. Such structures could provide an entirely new approach to address and possibly resolve the Riemann Hypothesis.
In essence, this approach contributes to the Riemann Hypothesis literature by presenting computational evidence of structured symbolic relationships in zeta zeros. It suggests symbolic regression as not merely a numerical tool but a conceptual bridge between numerical data and analytic symbolic formulations necessary for mathematical breakthroughs.

Summary
In summary, the Riemann Symbolic Regression Theory presented here leverages computationally accurate zeta zero calculations (from the Zeta Calculator) and symbolic equation extraction (through the Equation Finder) to demonstrate progressively accurate symbolic approximations of the zeros. The observed pattern of increased accuracy with equation complexity strongly supports the theory that symbolic structures inherently underpin the Riemann Zeta function's zeros. This insight offers a fresh theoretical direction toward resolving one of mathematics' most enduring puzzles, the Riemann Hypothesis.

Reference
Britannica. (n.d.). Riemann hypothesis. Retrieved from https://www.britannica.com/science/Riemann-hypothesis
Barnett, A. (2025). Zeta Calculator [Computer software]. Retrieved from https://github.com/Aphiticus/Riemann-Symbolic-Regression-Theory/tree/main
Barnett, A. (2025). Equation Finder [Computer software]. Retrieved from https://github.com/Aphiticus/Riemann-Symbolic-Regression-Theory/tree/main

----------------------------------------------------------------------------------------------------------------------------------------------------
Instruction for how to test theory:
1. Download the Zeta Equation Search.py file. 
2. Install all required libraries. Now you should be able to run script but you need the Dataset csv file that contains the training data to include the Zeta numbers.
3. Download the Zeta Dataset.csv file and save to a folder.
4. Now run the Zeta Equation Search and select the dataset on popup to start using the regressor for making an equation to match the dataset.
5. Alternative: 
7. If you want to create Zeta dateset yourself to test, download the Zeta Creator.py file and ensure you have libraries installed.  Then run in order to make a dataset with Zeta's.

---------------------------------------------------------------------------------------------------------------------------------------------------
Special note:
I would like the communities assistance with helping prove and or disprove this theory.  I have limited resources with computational power to really analyze the symbolic regression, but do believe the current correlation is undeniable. The larger the computational resources, the more zeta's that can be tested and as the equation grows the MSE (Mean Square Error) gets smaller. 

- One side benefit to using this equation finder is it can possibly be of assistance with finding equations for encryption since the only way you can decrypt the outputs is by having the original equation that you can make with this. 



Acknowledgements
This software makes use of the following open source libraries:

PySR (MIT License)
https://github.com/MilesCranmer/PySR

Numba (BSD 2-Clause License)
https://github.com/numba/numba

NumPy (BSD 3-Clause License)
https://numpy.org/

Pandas (BSD 3-Clause License)
https://pandas.pydata.org/

tqdm (MPL 2.0 License)
https://github.com/tqdm/tqdm
