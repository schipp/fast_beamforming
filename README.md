# Fast beamforming in Python

![beampowers](beampowers.png)

Cross-correlation beamforming can be realised in a few lines of Python matrix operations, making use of `pytorch` linear algebra optimisations for speed. For small problems (=smaller than memory), this is fast and efficient and fully parallel. For large problems, this approach fails, specifically when the matrices containing cross correlations become too large for memory. To solve this, we employ `dask` to divide the computations into multiple tasks than can run across arbitrary infrastructure while retaining much of the same syntax and logic.

We demonstrate this in the notebooks within this repository:

* `beamforming_pytorch.ipynb`: Beamforming for a small problem (based on `pytorch`)
* `beamforming_dask.ipynb`: Beamforming for a big problem (based on `dask`)

**Note on `dask`**

`dask` allows to employ the same algorithm and largely the same syntax as the `pytorch` version, which means one doesn't have to worry about developing a different algorithm that is not memory-limited. However, `dask` also introduces a new optimisation problem: The choice of "good" chunks sizes for the specific system at hand. This is specific to the compute infrastructure used. On the bright side, this has to be optimized only once for a given problem-geometry (number of stations, grid points, frequencies). Visit the [dask documentation](https://docs.dask.org/en/stable/understanding-performance.html) for more details.

## Background

### What is beamforming?

Beamforming is a phase-matching algorithm commonly used to estimate the origin and local phase velocity of a propagating wavefront. The most basic beamformer is the delay-and-sum beamformer, where recordings across an array of sensors are phase-shifted and summed (forming the beam) to test for the best beam, corresponding to best origin and velocity (Rost and Thomas, 2002).

### Cross-correlation beamforming

The cross-correlation beamformer (also Bartlett beamformer, conventional beamformer, etc.) applies the same delay-and-sum idea to correlation functions between all sensor pairs (Ruigrok et al. 2017). This has the major advantage that only the coherent part of the wavefield is taken into account. The major disadvantage is that the computation of cross correlations between all station pairs can become expensive fast, scaling with $n^2$.

A few different formulation of this beamformer exist. We write it in frequency domain as

$B = \sum_\omega \sum_j \sum_{k\neq j} K_{jk}(\omega) S_{kj}(\omega),$

where $B$ is the beampower, $K_{jk}(\omega) = d_j(\omega) d^H_k(\omega)$ the cross-spectral density matrix of recorded signals $d$, $S_{kj}(\omega) = s_j(\omega) s^H_k(\omega)$ the cross-spectral density matrix of synthetic signals $s$, $j$ and $k$ identify sensors, and $H$ the complex conjugate. We exclude auto-correlations $j=k$, because they contain no phase-information. Consequently, negative beampowers indicate anti-correlation.

The synthetic signals $s$ (often called replica vectors or Green's functions) are the expected wavefield for a chosen direction of arrival and velocity, most often in acoustic homogeneous half-space, $s_j = \exp(-i \omega t_j)$, where $t_j$ is the traveltime from source to each receiver $j$.

### Plane-wave beamforming

In seismology, "Beamforming" is often synonymous with plane-wave beamforming. In plane-wave beamforming $t_j$ is the travel time from a reference point (commonly center of array) and the sensor $j$ for a given plane-wave

$\boldsymbol{t_j} = \boldsymbol{r_j} \cdot \boldsymbol{u_h}$,

where $\boldsymbol{r_j} = (r_x, r_y)$ the coordinates of sensor $j$ relative to the reference point, and $\boldsymbol{u_h} = u_h(\sin(\Theta), \cos(\Theta))$ the horizontal slowness vector with $u_h$ the horizontal slowness and $\Theta$ the direction of arrival. $u_h$ and $\Theta$ are the parameters that are tested for (or equivalently $u_x, u_y$). Because plane waves are assumed, the source origin must be enough far away that the plane-wave assumption becomes adequate.

### Matched field processing

When curved wavefronts are assumed instead, sources may be located within the sensor array and the grid that is tested is defined in space instead of the slowness-domain. This is called Matched Field Processing (e.g., Baggeroer et al. 1988). In practice, the difference between plane-wave beamforming and matched field processing lies in the computation of the Green's functions $s_j$, or more precisely the expected traveltimes $t_j$.

In MFP, the travel time is computed as

$t_j = |\boldsymbol{r}_j - \boldsymbol{r}_s| / c$,

with $|\boldsymbol{r}_j - \boldsymbol{r}_s|$ the euclidean distance between sensor and source and $c$ the medium velocity. The parameters tested for in MFP are the source position $\boldsymbol{r}_s$ (1D, 2D, 3D) and, sometimes, the medium velocity $c$. A different name for MFP that is intuitive to seismologists may be curved-wave Beamforming.

The beamforming done is the notebooks here is Matched Field Processing.

### References

Rost, S. & Thomas, C., 2002. Array seismology: Methods and applications. *Reviews of Geophysics*, **40**, 2–1–2–27. doi:10.1029/2000RG000100

Ruigrok, E., Gibbons, S. & Wapenaar, K., 2017. Cross-correlation beamforming. *J Seismol*, **21**, 495–508. doi:10.1007/s10950-016-9612-6

Baggeroer, A.B., Kuperman, W.A. & Schmidt, H., 1988. Matched field processing: Source localization in correlated noise as an optimum parameter estimation problem. *The Journal of the Acoustical Society of America*, **83**, 571–587. doi:10.1121/1.396151