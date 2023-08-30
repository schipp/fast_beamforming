# Fast beamforming in Python

Cross-correlation beamforming can be realised in a few lines of Python matrix operations, making use of `pytorch` linear algebra optimisations for speed. For small problems (=smaller than memory), this is fast and efficient and fully parallel. For large problems, this approach fails, specifically when the matrices containing cross correlations become too large for memory. To solve this, we employ `dask` to divide the computations into multiple tasks than can run across arbitrary infrastructure while retaining much of the same syntax and logic.

We demonstrate this in the notebooks within this repository:

* `beamforming_pytorch.ipynb`: Beamforming for a small problem (based on `pytorch`)
* `beamforming_dask.ipynb`: Beamforming for a big problem (based on `dask`)

**Note on `dask`**

`dask` allows to employ the same algorithm and largely the same syntax as the `pytorch` version, which means one doesn't have to worry about developing a different algorithm that is not memory-limited. However, `dask` also introduces a new optimisation problem: The choice of "good" chunks sizes for the specific system at hand. This is specific to the compute infrastructure used. On the bright side, this has to be optimized only once for a given problem-geometry (number of stations, grid points, frequencies). Visit the [dask documentation](https://docs.dask.org/en/stable/understanding-performance.html) for more details.

## Background

### What is beamforming?

Beamforming is a phase-matching algorithm commonly used to estimate the direction of arrival and local phase velocity of a propagating wavefront. The most basic beamformer is the delay-and-sum beamformer, where recordings across an array of sensors are phase-shifted and summed (forming the beam) to test for the best beam, corresponding to best direction of arrival and velocity.

### Cross-correlation beamforming

The cross-correlation beamformer (also Bartlett beamformer, conventional beamformer, etc.) applies the same delay-and-sum idea to correlation functions between all sensor pairs. This has the major advantage that only the coherent part of the wavefield is taken into account. The major disadvantage is that the computation of cross correlations between all station pairs can become expensive fast, scaling with $n^2$.

A few different formulation of this beamformer exist. We write it in frequency domain as

$B = \sum_\omega \sum_j \sum_{k\neq j} K_{jk}(\omega) S_{kj}(\omega),$

where $B$ is the beampower, $K_{jk}(\omega) = d_j(\omega) d_k^*(\omega)$ the cross-spectral density matrix of recorded signals $d$, $S_{kj}(\omega) = s_j(\omega) s_k^*(\omega)$ the cross-spectral density matrix of synthetic signals $s$, and $j$ and $k$ identify sensors. We exclude auto-correlations $j=k$, because they contain no phase-information. Consequently, negative beampowers indicate anti-correlation.

The synthetic signals $s$ (often called replica vectors or Green's functions) are the expected wavefield for a chosen direction of arrival and velocity, most often in acoustic homogeneous half-space, $s_j = \exp(-i \omega t_j)$, where $t_j$ is the traveltime from source to each receiver $j$.

### Matched field processing

How the Green's function $s_j$, or more precisely the expected traveltime $t_j$, is computed, determines whether seismologists call this algorithm plane-wave Beamforming or Matched Field Processing (MFP).

In plane-wave beamforming $t_j$ is the travel time from a reference point (commonly center of array) and the sensor $j$ for a given plane-wave

$t_j = \mathbf{r}_j \cdot \mathbf{u}_{hor}$,

where $\mathbf{r}_j = (r_x, r_y)$ the coordinates of sensor $j$ relative to the reference point, and $\mathbf{u}_{hor} = u_{hor}(\sin(\Theta), \cos(\Theta))$ the horizontal slowness vector with $u_{hor}$ the horizontal slowness and $\Theta$ the direction of arrival. $u_{hor}$ and $\Theta$ are the parameters that are tested for.

In MFP, the travel time is computed as

$t_j = |\mathbf{r}_j - \mathbf{r}_s| / c$,

with $|\mathbf{r}_j - \mathbf{r}_s|$ the euclidean distance between sensor and source and $c$ the medium velocity. The parameters tested for in MFP are the source position $\mathbf{r}_s$ (1D, 2D, 3D) and the medium velocity $c$. Inherently, this allows curved wavefronts and source within a senesor array. A different name for MFP could be curved-wave Beamforming.

In the notebooks here, I show examples of Matched Field Processing.