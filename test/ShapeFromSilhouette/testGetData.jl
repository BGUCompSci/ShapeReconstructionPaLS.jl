using jInv.Mesh
using jInv.Utils
using jInv.ForwardShare
using ShapeReconstructionPaLS.Utils
using ParamLevelSet
using ShapeReconstructionPaLS.ShapeFromSilhouette;
using Statistics
using Distributed


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
sigmaH = getDefaultHeavySide();
u, = ParamLevelSetModelFunc(Mesh,m;computeJacobian=0,sigma = sigmaH);

theta_phi = 0.0*deg2rad.([37.0 24.0 ; 27.0 14.0;11.0 11.0]);
b = 0.0*[1.0 5.0 7.0 ; -2.3 5.2 3.5;-4.3 1.2 3.1]*mean(Mesh.h);


methods = [MATFree,RBFBasedSimple2,RBFBased];
D = Array{Array{Float32,2}}(undef,length(methods));

for k=1:1
	pFor = getSfSParam(Mesh,theta_phi,b,ScreenMeshX2X3,XScreenLoc,CamLoc,methods[k],nworkers());
	Dremote, = getData(u[:],pFor);
	D[k] = arrangeRemoteCallDataIntoLocalData(Dremote);
	println("Done!")
end

for k= 2
	pFor = getSfSParam(Mesh,theta_phi,b,ScreenMeshX2X3,XScreenLoc,CamLoc,methods[k],nworkers());
	Dremote, = getData(m[:],pFor);
	D[k] = arrangeRemoteCallDataIntoLocalData(Dremote);
	println("Done!")
end
k=3
m_wraped = wrapRBFparamAndRotationsTranslations(m,theta_phi,0.0*b)
pFor = getVisHullParam(Mesh,theta_phi,b,ScreenMeshX2X3,XScreenLoc,CamLoc,methods[k],nworkers());
Dremote, = getData(m_wraped[:],pFor);
D[k] = arrangeRemoteCallDataIntoLocalData(Dremote);

println("Done!")

# using PyPlot;
# using jInvVis
# close("all");
# figure();
# plotModel(reshape(u,32,32,32));
# figure(); imshow(reshape(D[1][:,1],tuple(ScreenMeshX2X3.n...)));
println("All these should be approx 0")
for k=2:length(D)
	r = vecnorm(D[k] - D[1])./vecnorm(D[1]);
	println(r);
	# figure(); imshow(reshape(D[k][:,1],tuple(ScreenMeshX2X3.n...)))
	if r > 1e-1
		println("Comparing 1 and ",k)
		# error("Get data is not the same in all methods");
	end
end

println("Testing Sensitivities:");

du = 0.1*randn(size(u));
for k=1
	pFor = getVisHullParam(Mesh,theta_phi,b,ScreenMeshX2X3,XScreenLoc,CamLoc,methods[k],1);
	Dremote,pFor = getData(u[:],pFor);
	Dk = arrangeRemoteCallDataIntoLocalData(Dremote);
	println("Done!")
	
	for ii = 1:8
		dm = (0.5^ii)*du;
		pFor_t = getVisHullParam(Mesh,theta_phi,b,ScreenMeshX2X3,XScreenLoc,CamLoc,methods[k],1);
		Dremote, = getData(u[:] + dm[:],pFor_t);
		Dk_t = arrangeRemoteCallDataIntoLocalData(Dremote);
		println("norm(dt-d0): ",norm(Dk_t[:]-Dk[:]),", norm(dt - d0 - J0*dm): ",norm(Dk_t[:] - Dk[:] - getSensMatVec(dm[:],u[:],fetch(pFor[1]))[:]));
	end
end
dmm = 0.05*randn(size(m));
for k = 2
	pFor = getVisHullParam(Mesh,theta_phi,b,ScreenMeshX2X3,XScreenLoc,CamLoc,methods[k],1);
	Dremote, = getData(m[:],pFor);
	Dk = arrangeRemoteCallDataIntoLocalData(Dremote);
	println("Done!")
	
	for ii = 1:8
		dm = (0.5^ii)*dmm;
		pFor_t = getVisHullParam(Mesh,theta_phi,b,ScreenMeshX2X3,XScreenLoc,CamLoc,methods[k],1);
		Dremote, = getData(m[:] + dm[:],pFor_t);
		Dk_t = arrangeRemoteCallDataIntoLocalData(Dremote);
		println("norm(dt-d0): ",norm(Dk_t[:]-Dk[:]),", norm(dt - d0 - J0*dm): ",norm(Dk_t[:] - Dk[:] - getSensMatVec(dm[:],m[:],fetch(pFor[1]))[:]));
	end
end
k=3
m_wraped = wrapRBFparamAndRotationsTranslations(m,theta_phi,0.0*b)
dm_wraped = wrapRBFparamAndRotationsTranslations(dmm,0.05*randn(size(theta_phi)),0.05*randn(size(b)));
pFor = getVisHullParam(Mesh,theta_phi,b,ScreenMeshX2X3,XScreenLoc,CamLoc,methods[k],1);
Dremote, = getData(m_wraped[:],pFor);
Dk = arrangeRemoteCallDataIntoLocalData(Dremote);
V = 0;
println("Done!")

for ii = 1:8
	dm = (0.5^ii)*dm_wraped;
	pFor_t = getVisHullParam(Mesh,theta_phi,b,ScreenMeshX2X3,XScreenLoc,CamLoc,methods[k],1);
	Dremote, = getData(m_wraped[:] + dm[:],pFor_t);
	Dk_t = arrangeRemoteCallDataIntoLocalData(Dremote);
	V = getSensMatVec(dm[:],m_wraped[:],fetch(pFor[1]))[:];
	println("norm(dt-d0): ",norm(Dk_t[:]-Dk[:]),", norm(dt - d0 - J0*dm): ",norm(Dk_t[:] - Dk[:] - V));
end







