import LinearAlgebra
using LinearAlgebra
using SparseArrays
using DelimitedFiles

using jInv
### matrix free S (slicing) matrix
export normalize

function normalize(xyz::Array{Float64},eps::Float64 = 1e-5)
	n = div(length(xyz),3);
	x = xyz[1:n];
	y = xyz[(n+1):2*n];
	z = xyz[(2*n+1):3*n];
	norms = sqrt.(x.^2 .+ y.^2 .+ z.^2 .+ eps);
	inv_n = 1.0./norms;
	inv_np3 = inv_n.^3;
	nx = x./norms;
	ny = y./norms;
	nz = z./norms;
	xyz_norm = [nx;ny;nz];
	#d(x/sqrt(x^2+y^2+z^2))_dx = 1/sqrt(x^2+y^2+z^2) - x^2/(x^2+y^2+z^2)^{3/2}
	#d(x/sqrt(x^2+y^2+z^2))_dy =  - x*y/(x^2+y^2+z^2)^{3/2}
	dxdx = sparse(Diagonal(inv_n .- (x.^2).*inv_np3));
	dxdy = sparse(Diagonal(      .- (x.*y).*inv_np3));
	dxdz = sparse(Diagonal(      .- (x.*z).*inv_np3));
	dydx = dxdy;
	dydy = sparse(Diagonal(inv_n .- (y.^2).*inv_np3));
	dydz = sparse(Diagonal( 	 .- (y.*z).*inv_np3));
	dzdx = dxdz;
	dzdy = dydz;
	dzdz = sparse(Diagonal(inv_n .- (z.^2).*inv_np3));
	J = [dxdx dxdy dxdz ; dydx dydy dzdy ; dzdx dzdy dzdz];
	return xyz_norm, J
end





export getData
function getData(m::Array,pFor::PointCloudParam,doClear::Bool=false)
d = 0;
npc = pFor.npcAll;
Parray = pFor.P;
n_points = 0;
for k=1:length(Parray)
	n_points+= size(Parray[k],1);
end
d = zeros(Float32,n_points);
Jacobian = convert(SparseMatrixCSC{Float64,Int32},spzeros(length(d),length(m)));

if pFor.method == MATFree	
	
	# global count = 1;
	# for i=1:npc
		# P = Parray[i];
		# dcurr = P*m;
		# d[count:(count + size(P,1) - 1)] = dcurr;
		# Jacobian[count:(count + size(P,1) - 1),:] = P;
		# count = count + size(P,1);
	# end 
	# pFor.Jacobian = Jacobian;
	
	u,JBuilder = MeshFreeParamLevelSetModelFunc(Mesh,m,computeJacobian = 0,sigma = sigmaH,bf = 1,numParamOfRBF = numParamOfRBF);
	
	return d,pFor;
end
	
if pFor.method == RBFBased || pFor.method == RBF10Based || pFor.method == RBF5Based
	if pFor.method == RBF10Based
		numParamOfRBF = 10;
	else
		numParamOfRBF = 5;
	end
	#nRBF 				= div(length(m)- 5*pFor.npcAll,numParamOfRBF) ;
	#(m1,theta_phi,b) 	= splitRBFparamAndRotationsTranslations(m,nRBF,pFor.npcAll,numParamOfRBF);
	#theta_phi 			= theta_phi[pFor.workerSubIdxs,:];
	#b 					= b[pFor.workerSubIdxs,:];
	#mrot,Jrot 			= rotateAndMoveRBF(m1,Minv,theta_phi,b;computeJacobian = 1,numParamOfRBF=numParamOfRBF);
	theta_phi = 0; b =0;
	d,JacT 				= getPCDataRBF(pFor, m,theta_phi,b,numParamOfRBF);
	nRBF 				= div(length(m),numParamOfRBF) ;
	# multiply Jacobian with Jrot, and then take the transpose.

	JacobianT = convert(SparseMatrixCSC{Float64,Int32},spzeros(length(m),prod(size(d))));
	IpIdxs = getIpIdxs(pFor.workerSubIdxs,nRBF,pFor.npcAll,numParamOfRBF);
	#JacobianT[IpIdxs,:] = Jrot'*JacT;
	#JacobianT[IpIdxs,:] = JacT;
	pFor.Jacobian = JacT';
end
return d,pFor;
end

function getPCDataRBF(pFor, m, theta_phi,b,numParamOfRBF = 5)
#Mesh = pFor.Mesh;
#n = pFor.Mesh.n;
n = [128,128,128];
Mesh = getRegularMesh([0.0 1.5 0.0 1.5 0.0 1.5],n);
ndips = pFor.npcAll;
npc = ndips;
#P is a sparse matrix of size k \times n, where k = #points on point cloud , n = volume size of the mesh
Jacobians = Array{SparseMatrixCSC{Float64,Int32}}(undef, npc);
count = 1;
Parray = pFor.P;
n_points = 0;
#for k=1:length(Parray)
#	n_points+= size(Parray[k],1);
#end
n_points = size(Parray,1);
d = zeros(Float32,n_points);
println("Parray size:",size(Parray))
println("npc in getdata:",npc)
npc =  1;
for i=1:npc
	sigmaH = getDefaultHeaviside();
	u,JBuilder = MeshFreeParamLevelSetModelFunc(Mesh,m,Xc = Parray,computeJacobian = 1,sigma = sigmaH,bf = 1,numParamOfRBF = numParamOfRBF);
	J1 = getSparseMatrixTransposed(JBuilder);
	P = Parray;
	#dcurr = P'*u;
	dcurr = u;
	#println("size u:",size(u))
	#d[count:(count + size(P,1) - 1)] = dcurr;
	d = dcurr;
	count = count + size(P,1);
	#println("J1 size:",size(J1));
	Jacobians[i] = (J1);
end
JacT = blockdiag(Jacobians...);
return d,JacT
end
