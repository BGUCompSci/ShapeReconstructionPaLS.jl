[![Build Status](https://travis-ci.com/BGUCompSci/ShapeReconstructionPaLS.jl.svg?branch=master)](https://travis-ci.com/BGUCompSci/ShapeReconstructionPaLS.jl)
[![Coverage Status](https://coveralls.io/repos/github/BGUCompSci/ShapeReconstructionPaLS.jl/badge.svg?branch=master)](https://coveralls.io/github/BGUCompSci/ShapeReconstructionPaLS.jl?branch=master)

# ShapeReconstructionPaLS
This is a flexible framework for multimodal 3D shape reconstruction using parametric level set methods. 

This package relies on two other packages:
1. [`jInv.jl`](https://github.com/JuliaInv/jInv.jl) - A juliA package to solve parameter estimation problems efficiently and in parallel  For more details see (http://arxiv.org/abs/1606.07399).
2. [`ParamLevelSet.jl`](https://github.com/JuliaInv/ParamLevelSet.jl) - A julia package to implement parametric level set methods.

# Overview

ShapeReconstructionPaLS consists of three submodules:

1. `DipTransform` - For shape reconstruction from dipping objects in water in different angles, exploiting Archimedes Law.
2. `ShapeFromSilhouette` - A no-fill implementation for surface reconstruction from silhouette with support to camera calibration fixup. 
3. `PointCloud` - For surface reconstruction from point clouds with support to registration fixup. 
4. `Direct` - A module to find a parametric level set representation for a given object. For demonstrating the expressiveness of the parametric level set representation.
5. `Utils` - utility functions

# Examples

Some examples can be found in the `driver` folder. See the results section in https://arxiv.org/abs/1904.10379.

# Acknowledgements


