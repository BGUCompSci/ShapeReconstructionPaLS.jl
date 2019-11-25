using jInv.Mesh
using jInv.Utils
using jInv.ForwardShare
using ShapeReconstructionPaLS.Utils
using ParamLevelSet
using ShapeReconstructionPaLS.ShapeFromSilhouette;
using Statistics
using Distributed
using LinearAlgebra
using SparseArrays
using Test

println("Test SoftMax")
n = 20;
m = 10;
u = rand(20);
At = sprand(n,m,4.0/m);
At.nzval[:] .= 1.0;


d0,Jt = softMaxProjWithSensMat(At,u);

du0 = 0.01*randn(size(u));

U = spdiagm(0 => u);
dataMax = maximum(At'*U,dims=2);
r = norm(dataMax .- d0)./norm(d0);
@test r < 0.025
for ii = 1:8
	du = (0.5^ii)*du0;
	ut = u + du;
	dt, = softMaxProjWithSensMat(At,ut);
	println("norm(dt-d0): ",norm(dt-d0),", norm(dt - d0 - J0*dm): ",norm(dt - d0 - Jt'*du));
end

