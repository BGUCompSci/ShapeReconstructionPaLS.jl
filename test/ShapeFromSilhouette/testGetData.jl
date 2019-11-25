using jInv.Mesh
using jInv.Utils
using jInv.ForwardShare
using ShapeReconstructionPaLS.Utils
using ParamLevelSet
using ShapeReconstructionPaLS.ShapeFromSilhouette;
using Statistics
using Distributed
using LinearAlgebra
using Test


println("Testing getData for all the methods:")
n = [32,32,32];
Mesh = getRegularMesh([0.0;3.0;0.0;3.0;0.0;3.0],n);
ScreenMeshX2X3 = getRegularMesh([0.0 3.0 0.0 3.0],n[2:3]);
XScreenLoc = 5.0;
CamLoc = [-10.0 ; 0.0 ;0.0];


alpha = [1.5;2.5;2.0;-1.0];
beta = [1.0;2.0;1.6;1.5];
Xs = [1.5 1.5 1.5 ; 2.0 2.0 2.0; 1.2 2.3 1.5; 2.2 1.5 2.0];
# alpha = [1.0;];
# beta = [1.0;];
# Xs = [1.5 1.5 1.5 ;];

m = wrapTheta(alpha,beta,Xs);
sigmaH = getDefaultHeaviside();
u, = ParamLevelSetModelFunc(Mesh,m;computeJacobian=0,sigma = sigmaH);

theta_phi = 0.0*deg2rad.([37.0 24.0 ; 27.0 14.0;11.0 11.0]);
b = 0.0*[1.0 5.0 7.0 ; -2.3 5.2 3.5;-4.3 1.2 3.1]*mean(Mesh.h);


methods = [MATFree,RBFBasedSimple2,RBFBased];
D = Array{Array{Float32,2}}(undef,length(methods));

for k=1:1
	println("test GetData MATFree");
	pFor = getSfSParam(Mesh,theta_phi,b,ScreenMeshX2X3,XScreenLoc,CamLoc,methods[k],nworkers());
	Dremote, = getData(u[:],pFor);
	D[k] = arrangeRemoteCallDataIntoLocalData(Dremote);
end

for k=2
	println("test GetData RBFBasedSimple2")
	pFor = getSfSParam(Mesh,theta_phi,b,ScreenMeshX2X3,XScreenLoc,CamLoc,methods[k],nworkers());
	Dremote, = getData(m[:],pFor);
	D[k] = arrangeRemoteCallDataIntoLocalData(Dremote);
end

k=3
println("test GetData RBFBased")
m_wraped = wrapRBFparamAndRotationsTranslations(m,theta_phi,0.0*b)
pFor = getSfSParam(Mesh,theta_phi,b,ScreenMeshX2X3,XScreenLoc,CamLoc,methods[k],nworkers());
Dremote, = getData(m_wraped[:],pFor);
D[k] = arrangeRemoteCallDataIntoLocalData(Dremote);
println("Done!")

# using PyPlot;
# using jInvVisPyPlot
# close("all");
# figure();
# plotModel(reshape(u,32,32,32));
# figure(); imshow(reshape(D[1][:,1],tuple(ScreenMeshX2X3.n...)));

for k=2:length(D)
	r = norm(D[k] - D[1])./norm(D[1]);
	# figure(); imshow(reshape(D[k][:,1],tuple(ScreenMeshX2X3.n...)))
	println("Comparing 1 and ",k,". Relative diff: ",r);
	@test r <= 1.0
end

println("Testing Sensitivities: MATFree");
du = 0.01*randn(size(u));
for k=1
	pFor = getSfSParam(Mesh,theta_phi,b,ScreenMeshX2X3,XScreenLoc,CamLoc,methods[k],1);
	Dremote,pFor = getData(u[:],pFor);
	pFor = fetch(pFor[1]);
	Dk = arrangeRemoteCallDataIntoLocalData(Dremote);
	for ii = 1:8
		dm = (0.5^ii)*du;
		pFor_t = getSfSParam(Mesh,theta_phi,b,ScreenMeshX2X3,XScreenLoc,CamLoc,methods[k],1);
		Dremote, = getData(u[:] + dm[:],pFor_t);
		Dk_t = arrangeRemoteCallDataIntoLocalData(Dremote);
		println("norm(dt-d0): ",norm(Dk_t[:]-Dk[:]),", norm(dt - d0 - J0*dm): ",norm(Dk_t[:] .- Dk[:] .- getSensMatVec(dm[:],u[:],pFor)[:]));
	end
end
println("Testing Sensitivities: RBFBasedSimple2");
dmm = 0.01*randn(size(m));
for k = 2
	pFor = getSfSParam(Mesh,theta_phi,b,ScreenMeshX2X3,XScreenLoc,CamLoc,methods[k],1);
	Dremote, = getData(m[:],pFor);
	pFor = fetch(pFor[1]);
	Dk = arrangeRemoteCallDataIntoLocalData(Dremote);
	
	for ii = 1:8
		dm = (0.5^ii)*dmm;
		pFor_t = getSfSParam(Mesh,theta_phi,b,ScreenMeshX2X3,XScreenLoc,CamLoc,methods[k],1);
		Dremote, = getData(m[:] + dm[:],pFor_t);
		Dk_t = arrangeRemoteCallDataIntoLocalData(Dremote);
		println("norm(dt-d0): ",norm(Dk_t[:].-Dk[:]),", norm(dt - d0 - J0*dm): ",norm(Dk_t[:] .- Dk[:] .- getSensMatVec(dm[:],m[:],pFor)[:]));
	end
end
println("Testing Sensitivities: RBFBased");
k=3
m_wraped = wrapRBFparamAndRotationsTranslations(m,theta_phi,0.0*b)
dm_wraped = wrapRBFparamAndRotationsTranslations(dmm,0.05*randn(size(theta_phi)),0.05*randn(size(b)));
pFor = getSfSParam(Mesh,theta_phi,b,ScreenMeshX2X3,XScreenLoc,CamLoc,methods[k],1);
Dremote, = getData(m_wraped[:],pFor);
Dk = arrangeRemoteCallDataIntoLocalData(Dremote);
V = 0;


for ii = 1:8
	dm = (0.5^ii)*dm_wraped;
	pFor_t = getSfSParam(Mesh,theta_phi,b,ScreenMeshX2X3,XScreenLoc,CamLoc,methods[k],1);
	Dremote, = getData(m_wraped[:] + dm[:],pFor_t);
	Dk_t = arrangeRemoteCallDataIntoLocalData(Dremote);
	V = getSensMatVec(dm[:],m_wraped[:],fetch(pFor[1]))[:];
	println("norm(dt-d0): ",norm(Dk_t[:]-Dk[:]),", norm(dt - d0 - J0*dm): ",norm(Dk_t[:] - Dk[:] - V));
end







