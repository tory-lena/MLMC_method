### Multi-Level Monte Carlo (MLMC) Method

The two main method files MLMC.jl and MLMC2.jl provide a julia implementation of a multi-level monte carlo. MLMC is a recently developed approach, which leverages the idea of control variates for variance reduction and reducing computational cost by performing more low cost in combination with a few high accuracy simulations. The remaining files are examples taken from different fields. To learn more about the method itself feel free to download the pdf or stop by this [webpage] by Prof. Mike Giles.

#### How to use it?
Download the files and examples you wish to try
```julia
using Polynomials
using PyPlot
include("MLMC.jl")
```
Or include "MLMC2.jl" method. Then to test an example:

```julia
include("para.jl")
para()
```

#### Available Example

Parabolic <br/>
Elliptic <br/>
Navier Stokes <br/>
Thermodynamic Motions (salt.jl) <br/>
Reflected waves <br/>
Population dynamics <br/>
Basket Options (opre.jl) <br/>

Some fucntions use MLMC.jl others requires MLMC2.jl - the functions will handle to choose the apporiate implenetation.


#### Nomenclature in the files
```matlab
% P     = value 
% Nl    = number of samples at each level 
% cost  = total cost
%
% N0    = initial number of samples         > 0
% eps   = desired accuracy (rms error)      > 0 
% Lmin  = minimum level of refinement       >= 2
% Lmax  = maximum level of refinement       >= Lmin
%
% alpha -> weak error is  O(2^{-alpha*l})
% beta  -> variance is    O(2^{-beta*l})
% gamma -> sample cost is O(2^{gamma*l})
%
% varargin = optional additional user variables to be passed to mlmc_l
%
% if alpha, beta, gamma are not positive, then they will be estimated
%
% mlmc_l = function for level l estimator 
%
% [sums, cost] = mlmc_fn(l,N, varargin)     low-level routine
%
% inputs:  l = level
%          N = number of samples
%          varargin = optional additional user variables
%
% output: sums(1) = sum(Y)
%         sums(2) = sum(Y.^2)
%         cost = cost of N samples
```

[webpage]: https://people.maths.ox.ac.uk/gilesm/
