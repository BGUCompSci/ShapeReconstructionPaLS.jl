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
Minv = pFor.Mesh;
n = Minv.n;
npc = pFor.npcAll;
samplingBinning = pFor.samplingBinning;
ind_array = pFor.P;
n_points = 0;
for k=1:length(ind_array)
	n_points+=length(ind_array[k]);
end
d = zeros(Float32,3*n_points);
Jacobian = convert(SparseMatrixCSC{Float64,Int32},spzeros(length(d),length(m)));

if pFor.method == MATFree	
	Af   = getFaceAverageMatrix(Minv)
	A1 = Af[:,1:Int(size(Af,2)/3)];
	A2 = Af[:,Int(size(Af,2)/3)+1:2*Int(size(Af,2)/3)];
	A3 = Af[:,2*Int(size(Af,2)/3)+1:Int(size(Af,2))];
	Af = blockdiag(A1,A2,A3);
	
	D1 = kron(sparse(1.0I, Minv.n[3], Minv.n[3]),kron(sparse(1.0I, Minv.n[2], Minv.n[2]),ddx(Minv.n[1])))
	D2 = kron(sparse(1.0I, Minv.n[3], Minv.n[3]),kron(ddx(Minv.n[2]),sparse(1.0I, Minv.n[1], Minv.n[1])))
	D3 = kron(ddx(Minv.n[3]),kron(sparse(1.0I, Minv.n[2], Minv.n[2]),sparse(1.0I, Minv.n[1], Minv.n[1])))
	D = [D1 D2 D3];
	global count = 1;
	for i=1:npc
		ind = ind_array[i];
		nx = A1*(D1'*m);
		ny = A2*(D2'*m);
		nz = A3*(D3'*m);
		nx = nx[ind];
		ny = ny[ind];
		nz = nz[ind];
		#normals = Af*D*m;
		#normals,J = normalize([nx;ny;nz]);
		normals = [nx;ny;nz];
		d[count:(count + 3*length(ind) - 1)] = normals;
		I2 = collect(1:length(ind));
		J2 = ind;
		V2 = ones(size(ind));
		P = sparse(I2,J2,V2,length(ind),length(m));
		P = blockdiag(P,P,P);
		#J = J*P*Af*D';
		J = P*Af*D';
		Jacobian[count:(count + 3*length(ind) - 1),:] = J;
		count = count + 3*length(ind);
	end 
	pFor.Jacobian = Jacobian;
	return d,pFor;
end

if pFor.method == RBFBasedSimple || pFor.method == RBF10BasedSimple
	if pFor.method == RBF10BasedSimple
		numParamOfRBF = 10;
	else
		numParamOfRBF = 5;
	end
        println("numparam:",numParamOfRBF);
	println("m size:",size(m));
	println("m:",m)
	theta_phi = pFor.theta_phi_rad;
	b = pFor.b;
	println("theta phi:",theta_phi); println("b:",b)
	mrot,Jrot = rotateAndMoveRBFsimple(m,pFor.Mesh,theta_phi,b;computeJacobian = 1,numParamOfRBF=numParamOfRBF);
	println("mrot size:",size(mrot));
	println("mrot:",mrot)
	d,JacT = getPCDataRBF(pFor, mrot,theta_phi,b,numParamOfRBF);
	# multiply Jacobian with Jrot, and then take the transpose.
	pFor.Jacobian = Jrot'*JacT;
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
	nRBF = div(length(m),numParamOfRBF) ;
	println("size of m:",size(m));
	d,JacT 				= getPCDataRBF(pFor, m,theta_phi,b,numParamOfRBF);
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
Mesh = pFor.Mesh;
n = pFor.Mesh.n;
h = pFor.Mesh.h;
ndips = pFor.npcAll;
volCell = prod(Mesh.h);
npc = ndips;
eps = 1e-4/Mesh.h[1];
Af   = getFaceAverageMatrix(Mesh)
A1 = Af[:,1:Int(size(Af,2)/3)];
A2 = Af[:,Int(size(Af,2)/3)+1:2*Int(size(Af,2)/3)];
A3 = Af[:,2*Int(size(Af,2)/3)+1:Int(size(Af,2))];
Af = blockdiag(A1,A2,A3);


D1 = kron(sparse(1.0I, Mesh.n[3], Mesh.n[3]),kron(sparse(1.0I, Mesh.n[2], Mesh.n[2]),ddx(Mesh.n[1])))
D2 = kron(sparse(1.0I, Mesh.n[3], Mesh.n[3]),kron(ddx(Mesh.n[2]),sparse(1.0I, Mesh.n[1], Mesh.n[1])))
D3 = kron(ddx(Mesh.n[3]),kron(sparse(1.0I, Mesh.n[2], Mesh.n[2]),sparse(1.0I, Mesh.n[1], Mesh.n[1])))
D = [D1 D2 D3];

#P is a sparse matrix of size k \times n, where k = #points on point cloud , n = volume size of the mesh
Jacobians = Array{SparseMatrixCSC{Float64,Int32}}(undef, npc);
count = 1;
ind_array = pFor.P;

n_points = 0;
for k=1:length(ind_array)
	n_points+=length(ind_array[k]);
end
d = zeros(Float32,3*n_points);

for i=1:npc
	sigmaH = getDefaultHeavySide();
	u,I1,J1,V1 = ParamLevelSetModelFunc(Mesh,m[:,i];computeJacobian = 1,sigma = sigmaH,bf = 1,numParamOfRBF = numParamOfRBF);
	J1 = sparse(I1,J1,V1,prod(n),length(m[:,i]));
	#u,I1,J1,V1 = ParamLevelSetModelFunc(Mesh,m;computeJacobian = 1,sigma = sigmaH,bf = 1,numParamOfRBF = numParamOfRBF);
	#J1 = sparse(I1,J1,V1,prod(n),length(m));
	ind = ind_array[i];
	nx = A1*(D1'*u);
	ny = A2*(D2'*u);
	nz = A3*(D3'*u);
	nx = nx[ind];
	ny = ny[ind];
	nz = nz[ind];
	#normals = Af*D*m;
	#normals,J = normalize([nx;ny;nz]);
	normals = [nx;ny;nz];
	d[count:(count + 3*length(ind) - 1)] = normals;
	I2 = collect(1:length(ind));
	J2 = ind;
	V2 = ones(size(ind));
	P = sparse(I2,J2,V2,length(ind),length(u));
	P = blockdiag(P,P,P);
	#J = J*P*Af*D';
	J = P*Af*D';
	#Jacobian[count:(count + 3*length(ind) - 1),:] = J1'*J';
	count = count + 3*length(ind);
	Jacobians[i] = J1'*J';
	
end
JacT = blockdiag(Jacobians...);
return d,JacT
end
