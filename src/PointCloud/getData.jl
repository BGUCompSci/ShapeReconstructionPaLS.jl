import LinearAlgebra
using LinearAlgebra
using SparseArrays
using DelimitedFiles

using jInv
### matrix free S (slicing) matrix
export normalize
export getData
function getData(m::Array,pFor::PointCloudParam,doClear::Bool=false)
d = 0;
npc = pFor.npcAll;
Parray = pFor.P;
n_points = 0;
#for k=1:length(Parray)
#	n_points+= size(Parray[k],1);
#end
n_points = size(Parray,1);
d = zeros(Float32,n_points);
Jacobian = convert(SparseMatrixCSC{Float64,Int32},spzeros(length(d),length(m)));

if pFor.method == MATFree	
	n = [128,128,128];
	Mesh = getRegularMesh([0.0 3.0 0.0 3.0 0.0 3.0],n);
	if pFor.method == RBF10Based
		numParamOfRBF = 10;
	else
		numParamOfRBF = 5;
	end
	# global count = 1;
	# for i=1:npc
		# P = Parray[i];
		# dcurr = P*m;
		# d[count:(count + size(P,1) - 1)] = dcurr;
		# Jacobian[count:(count + size(P,1) - 1),:] = P;
		# count = count + size(P,1);
	# end 
	# pFor.Jacobian = Jacobian;
	
	Jacobian = convert(SparseMatrixCSC{Float64,Int32},Matrix(1.0I,length(m),length(m)));
	pFor.Jacobian = Jacobian;
	println("size of u:",size(m)); println("size of jacobian:",size(Jacobian));
	d = m;
	return d,pFor;
end
	
if pFor.method == RBFBased || pFor.method == RBF10Based || pFor.method == RBF5Based
	n = [128,128,128];
	Minv = getRegularMesh([0.0 3.0 0.0 3.0 0.0 3.0],n);
	if pFor.method == RBF10Based
		numParamOfRBF = 10;
	else
		numParamOfRBF = 5;
	end
	nRBF 				= div(length(m)- 5*pFor.npcAll,numParamOfRBF);
	(m1,theta_phi,b) 	= splitRBFparamAndRotationsTranslations(m,nRBF,pFor.npcAll,numParamOfRBF);
	theta_phi 			= theta_phi[pFor.workerSubIdxs,:];
	b 					= b[pFor.workerSubIdxs,:];
	mrot,Jrot 			= rotateAndMoveRBF(m1,Minv,theta_phi,b;computeJacobian = 1,numParamOfRBF=numParamOfRBF);
	d,JacT 				= getPCDataRBF(pFor,mrot,theta_phi,b,numParamOfRBF);
	nRBF 				= div(length(m),numParamOfRBF) ;
	# multiply Jacobian with Jrot, and then take the transpose.

	JacobianT = convert(SparseMatrixCSC{Float64,Int32},spzeros(length(m),prod(size(d))));
	IpIdxs = getIpIdxs(pFor.workerSubIdxs,nRBF,pFor.npcAll,numParamOfRBF);
	
	JacobianT = Jrot'*JacT;
	#JacobianT[IpIdxs,:] = JacT;
	pFor.Jacobian = JacobianT';
end
return d,pFor;
end

function getPCDataRBF(pFor, m, theta_phi,b,numParamOfRBF = 5)
#Mesh = pFor.Mesh;
#n = pFor.Mesh.n;
n = [128,128,128];
Mesh = getRegularMesh([0.0 3.0 0.0 3.0 0.0 3.0],n);
npc = pFor.npcAll;
#P is a sparse matrix of size k \times n, where k = #points on point cloud , n = volume size of the mesh
Jacobians = Array{SparseMatrixCSC{Float64,Int32}}(undef, npc);
count = 1;
Parray = pFor.P;
n_points = 0;
for k=1:length(Parray)
	n_points+= size(Parray[k],1);
end
d = zeros(Float32,n_points);
for i=1:npc
	P = Parray[i];
	sigmaH = getDefaultHeaviside();
	u,JBuilder = MeshFreeParamLevelSetModelFunc(Mesh,m[:,i],Xc = P,computeJacobian = 1,sigma = sigmaH,bf = 1,numParamOfRBF = numParamOfRBF);
	J1 = getSparseMatrixTransposed(JBuilder);
	#dcurr = P'*u;
	dcurr = u;
	d[count:(count + size(P,1) - 1)] = dcurr;
	count = count + size(P,1);
	Jacobians[i] = (J1);
end
JacT = blockdiag(Jacobians...);
return d,JacT
end
