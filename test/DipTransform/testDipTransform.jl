using jInv.Mesh
using jInv.Utils
using jInv.ForwardShare
using ShapeReconstructionPaLS.Utils
using ParamLevelSet
using ShapeReconstructionPaLS.DipTransform;
using Statistics
using Distributed
using LinearAlgebra
using Test


println("Testing getData for all the methods:")
n = [32,32,32];
Mesh = getRegularMesh([0.0;3.0;0.0;3.0;0.0;3.0],n);
alpha = [1.5;2.5;2.0;-1.0];
beta = [1.0;2.0;-1.6;1.5];
Xs = [1.5 1.5 1.5 ; 2.0 2.0 2.0; 1.2 2.3 1.5; 2.2 1.5 2.0];
m = wrapTheta(alpha,beta,Xs);
sigmaH = getDefaultHeaviside();
u, = ParamLevelSetModelFunc(Mesh,m;computeJacobian=0,sigma = sigmaH);
samplingBinning = 2;

theta_phi = deg2rad.([37.0 24.0 ; 27.0 14.0;11.0 11.0]);
b = 4.0*[1.0 2.0 3.0 ; -2.3 1.2 2.5;-1.3 1.2 1.1]*mean(Mesh.h);
b[:] .= 0.0;


# RBFBased        = "RBFs";

methods = [MATBased,MATFree,RBFBasedSimple1,RBFBasedSimple2,RBFBased];
D = Array{Array{Float32,2}}(undef,length(methods));

for k=1:2
	pFor = getDipParam(Mesh,theta_phi,b,samplingBinning,nworkers(),methods[k]);
	Dremote, = getData(u[:],pFor);
	D[k] = arrangeRemoteCallDataIntoLocalData(Dremote);
end

for k=3:4
	pFor = getDipParam(Mesh,theta_phi,b,samplingBinning,nworkers(),methods[k]);
	Dremote, = getData(m[:],pFor);
	D[k] = arrangeRemoteCallDataIntoLocalData(Dremote);
end
k=5
m_wraped = wrapRBFparamAndRotationsTranslations(m,theta_phi,b)
pFor = getDipParam(Mesh,theta_phi,b,samplingBinning,nworkers(),methods[k]);
Dremote, = getData(m_wraped[:],pFor);
D[k] = arrangeRemoteCallDataIntoLocalData(Dremote);

println("All these should be approx 0")
for k=3:5
	r = norm(D[k] - D[2])./norm(D[2]);
	if r > 5e-1
		println("testGetData: Get data is not the same in all methods");
	end
	@test r < 5e-1;
end


println("Testing Sensitivities:");

du = 0.1*randn(size(u));
for k=1:2
	pFor = getDipParam(Mesh,theta_phi,b,samplingBinning,1,methods[k]);
	Dremote,pFor = getData(u[:],pFor);
	Dk = arrangeRemoteCallDataIntoLocalData(Dremote);
	
	ui = u; # not sure why this is needed. Otherwise julia do not recognise u in the loop
	for ii = 1:3
		dm = (0.5^ii)*du;
		println("Testing sensitivities for Method ",methods[k])
		pFor_t = getDipParam(Mesh,theta_phi,b,samplingBinning,1,methods[k]);
		
		Dremote, = getData(ui[:] .+ dm[:],pFor_t);
		Dk_t = arrangeRemoteCallDataIntoLocalData(Dremote);
		println("norm(dt-d0): ",norm(Dk_t[:].-Dk[:]),", norm(dt - d0 - J0*dm): ",norm(Dk_t[:] .- Dk[:] .- getSensMatVec(dm[:],u[:],fetch(pFor[1]))[:]));
		
		println("----------- Transpose test --------------");
		
		ui = reshape(ui,tuple(n...));
		u1 = copy(ui);
		d1 = sampleSlices(ui,0.01,1);
		t1 = randn(size(d1));
		sampleSlicesT(t1,0.01,u1);
		@test abs(dot(t1,d1) - dot(u1,ui)) < 1e-5	
		
		Vd = randn(size(Dk_t));
		vm = randn(size(dm));
		val1 = dot(getSensMatVec(vm,ui[:],fetch(pFor[1]))[:],Vd);
		val2 = dot(getSensTMatVec(Vd[:],ui[:],fetch(pFor[1]))[:],vm);
		println(methods[k])
		@test abs(val1-val2) < 1e-5	
	end
end
dmm = 0.05*randn(size(m));
for k=3:4
	println("Testing sensitivities for Method ",methods[k])
	pFor = getDipParam(Mesh,theta_phi,b,samplingBinning,1,methods[k]);
	Dremote, = getData(m[:],pFor);
	Dk = arrangeRemoteCallDataIntoLocalData(Dremote);
	println("Done!")
	
	for ii = 1:8
		dm = (0.5^ii)*dmm;
		pFor_t = getDipParam(Mesh,theta_phi,b,samplingBinning,1,methods[k]);
		Dremote, = getData(m[:] + dm[:],pFor_t);
		Dk_t = arrangeRemoteCallDataIntoLocalData(Dremote);
		println("norm(dt-d0): ",norm(Dk_t[:]-Dk[:]),", norm(dt - d0 - J0*dm): ",norm(Dk_t[:] - Dk[:] - getSensMatVec(dm[:],m[:],fetch(pFor[1]))[:]));
	end
end
k=5
println("Testing sensitivities for Method ",methods[k])
m_wraped = wrapRBFparamAndRotationsTranslations(m,theta_phi,b)
dm_wraped = wrapRBFparamAndRotationsTranslations(dmm,0.05*randn(size(theta_phi)),0.05*randn(size(b)));
pFor = getDipParam(Mesh,theta_phi,b,samplingBinning,nworkers(),methods[k]);
Dremote, = getData(m_wraped[:],pFor);
Dk = arrangeRemoteCallDataIntoLocalData(Dremote);
V = 0;
println("Done!")

for ii = 1:8
	dm = (0.5^ii)*dm_wraped;
	pFor_t = getDipParam(Mesh,theta_phi,b,samplingBinning,1,methods[k]);
	Dremote, = getData(m_wraped[:] + dm[:],pFor_t);
	Dk_t = arrangeRemoteCallDataIntoLocalData(Dremote);
	V = getSensMatVec(dm[:],m_wraped[:],fetch(pFor[1]))[:];
	println("norm(dt-d0): ",norm(Dk_t[:]-Dk[:]),", norm(dt - d0 - J0*dm): ",norm(Dk_t[:] - Dk[:] - V));
end







