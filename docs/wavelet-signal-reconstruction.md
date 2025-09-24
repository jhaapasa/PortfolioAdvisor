Yes, you only need **a single pass of MODWT** to level J.

A single MODWT decomposition provides all the necessary components to reconstruct the smooth signals for every level from 1 to J. You don't need to run the transform multiple times.

***

## How It Works

When you perform a MODWT on your price signal to a maximum level $J$, it breaks the signal down into a set of coefficients. For a signal $X(t)$, the decomposition yields:

* **J sets of detail coefficients:** $W_1, W_2, ..., W_J$
* **One set of final smooth coefficients:** $V_J$

These coefficients represent different timescales within your data. $W_1$ captures the finest, highest-frequency details, while $V_J$ captures the coarsest, lowest-frequency trend.

The original signal can be perfectly reconstructed by summing the individual signals derived from these coefficients:

$$X(t) = D_1(t) + D_2(t) + ... + D_J(t) + S_J(t)$$

Where:
* $D_j(t)$ is the **detail signal** at level $j$, reconstructed from the detail coefficients $W_j$.
* $S_J(t)$ is the **final smooth signal** (the main trend) at level $J$, reconstructed from the smooth coefficients $V_J$.



## Reconstructing All Smooth Signals

From this single decomposition, you can easily construct the smooth signal for *any* level $j$ between 1 and J.

The smooth signal at a given level, $S_j$, is simply the sum of the final smooth signal ($S_J$) and all the detail signals at coarser scales (i.e., levels higher than $j$).

The formulas are:

* **Final Smooth Signal (Level J):** This is the smoothest component, directly obtained by applying the inverse MODWT to the $V_J$ coefficients.
    $$S_J(t)$$

* **Intermediate Smooth Signals (Level j < J):** To get the smooth signal for any other level $j$, you add the details back in, starting from the level above it.
    $$S_j(t) = S_{j+1}(t) + D_{j+1}(t)$$
    Or more generally:
    $$S_j(t) = S_J(t) + \sum_{k=j+1}^{J} D_k(t)$$

For example, to get $S_1$ (the least smooth "smooth" signal, which is just the original signal minus the finest details), you would calculate:

$$S_1(t) = S_J(t) + D_2(t) + D_3(t) + ... + D_J(t)$$

Therefore, one pass to your maximum desired level $J$ gives you all the building blocks ($D_1, ..., D_J, S_J$) needed to assemble and plot any smooth signal $S_1, ..., S_J$.