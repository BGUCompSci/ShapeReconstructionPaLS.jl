import LinearAlgebra
using LinearAlgebra
using SparseArrays
using DelimitedFiles

using jInv
### matrix free S (slicing) matrix


function subv2ind(shape, cindices)
    """Return linear indices given vector of cartesian indices.
    shape: d-tuple of dimensions.
    cindices: n-iterable of d-tuples, each containing cartesian indices.
    Returns: n-vector of linear indices.

    Based on:
    https://discourse.julialang.org/t/psa-replacement-of-ind2sub-sub2ind-in-julia-0-7/14666/8
    Similar to this matlab function:
    https://github.com/tminka/lightspeed/blob/master/subv2ind.m
    """
    lndx = LinearIndices(Dims(shape))
    n = length(cindices)
    out = Array{Int}(undef, n)
    for i = 1:n
        out[i] = lndx[cindices[i]...]
    end
    return out
end

export getData
function getData(m::Array,pFor::PointCloudParam,doClear::Bool=false)
d = 0;
Mesh = pFor.Mesh;
n = Mesh.n;
ndips = 1;
samplingBinning = pFor.samplingBinning;

if pFor.method == MATFree	
	#Data:
	ind = pFor.P;
	nx = pFor.Normals[:,1]; ny = pFor.Normals[:,2]; nz = pFor.Normals[:,3];
	margin = pFor.margin;
	npc = pFor.npcAll;
	subs = ind2subv(pFor.Mesh.n,ind);
	Jacobians = Array{SparseMatrixCSC{Float64,Int32}}(undef, 2);

	
	subsP1 = subs[1:round(Int,(0.5+margin)*length(subs))];
	subsP2 = subs[round(Int,(0.5-margin)*length(subs)):length(subs)];
	if(length(subsP2) != length(subsP1))
		println("point clouds size mismatch");
	end
	d = zeros(Float32,length(subsP1)*3,npc);
	for i=1:npc
		cursubs = subs[round(Int,(i-1)*(1/npc - margin)*length(subs))+1:min(length(subs),round(Int,i*(1/npc + margin)*length(subs)))];
		println("cursubs size:",size(cursubs));
		b = tuple(pFor.b[i,:]...);
		for ii = 1:length(cursubs)
			cursubs[ii] = round.(Int,cursubs[ii] .+ b);
		end
		
		ind = subv2ind(pFor.Mesh.n,cursubs);
		
		I2 = collect(1:length(ind));
		J2 = ind;
		V2 = ones(size(ind));
		P = sparse(I2,J2,V2,length(ind),length(m));
		
		
		nxt = P*nx; nyt = P*ny; nzt = P*nz;
		curnormals = [nxt nyt nzt];
		d[:,i] = curnormals[:];
		
	end
	
	println("simulated data size:",size(d));
	ind = pFor.P;
	println("ind size:",size(ind))
	I2 = ind;
	J2 = ind;
	V2 = ones(size(ind));
	Pp = sparse(I2,J2,V2,length(m),length(m));
	
	
	Mesh = pFor.Mesh;
	n = pFor.Mesh.n;
	h = pFor.Mesh.h;
	ndips = 2;
	volCell = prod(Mesh.h);
	pFor.Jacobian = (Pp*Af*Div')';
	
	println("size of jac",size(pFor.Jacobian));
	return d,pFor;
	end
	
if pFor.method == RBFBased || pFor.method == RBF10Based || pFor.method == RBF5Based
	if pFor.method == RBF10Based
		numParamOfRBF = 10;
	else
		numParamOfRBF = 5;
	end
	nRBF 				= div(length(m)- 5*pFor.npcAll,numParamOfRBF) ;
	(m1,theta_phi,b) 	= splitRBFparamAndRotationsTranslations(m,nRBF,pFor.npcAll,numParamOfRBF);
	theta_phi 			= theta_phi[pFor.workerSubIdxs,:];
	b 					= b[pFor.workerSubIdxs,:];
	mrot,Jrot 			= rotateAndMoveRBF(m1,Mesh,theta_phi,b;computeJacobian = 1,numParamOfRBF=numParamOfRBF);
	println("mrot size:",size(mrot));
	d,JacT 				= getPCDataRBF(pFor, mrot,theta_phi,b,numParamOfRBF);
	# multiply Jacobian with Jrot, and then take the transpose.

	JacobianT = convert(SparseMatrixCSC{Float64,Int32},spzeros(length(m),prod(size(d))));
	IpIdxs = getIpIdxs(pFor.workerSubIdxs,nRBF,pFor.npcAll,numParamOfRBF);
	JacobianT[IpIdxs,:] = Jrot'*JacT;
	pFor.Jacobian = JacobianT';
end
return d,pFor;
end

function getPCDataRBF(pFor, m, theta_phi,b,numParamOfRBF = 5)
Mesh = pFor.Mesh;
n = pFor.Mesh.n;
h = pFor.Mesh.h;
ndips = 2;
volCell = prod(Mesh.h);

eps = 1e-4/Mesh.h[1];
Af   = getFaceAverageMatrix(Mesh)
A1 = Af[:,1:Int(size(Af,2)/3)]; println("A1 size:",size(A1))
A2 = Af[:,Int(size(Af,2)/3)+1:2*Int(size(Af,2)/3)];
A3 = Af[:,2*Int(size(Af,2)/3)+1:Int(size(Af,2))];


D1 = kron(sparse(1.0I, Mesh.n[3], Mesh.n[3]),kron(sparse(1.0I, Mesh.n[2], Mesh.n[2]),ddx(Mesh.n[1])))
D2 = kron(sparse(1.0I, Mesh.n[3], Mesh.n[3]),kron(ddx(Mesh.n[2]),sparse(1.0I, Mesh.n[1], Mesh.n[1])))
D3 = kron(ddx(Mesh.n[3]),kron(sparse(1.0I, Mesh.n[2], Mesh.n[2]),sparse(1.0I, Mesh.n[1], Mesh.n[1])))


#P is a sparse matrix of size k \times n, where k = #points on point cloud , n = volume size of the mesh
Jacobians = Array{SparseMatrixCSC{Float64,Int32}}(undef, 2);
indices = pFor.P;
I2 = collect(1:length(indices));
J2 = indices;
V2 = ones(size(indices));
P = sparse(I2,J2,V2,length(indices),prod(Mesh.n));


full_ind = pFor.P;
margin = pFor.margin;
npc = pFor.npcAll;
subs = ind2subv(pFor.Mesh.n,full_ind);
subsP1 = subs[1:round(Int,(0.5+margin)*length(subs))];
subsP2 = subs[round(Int,(0.5-margin)*length(subs)):length(subs)];

if(length(subsP2) != length(subsP1))
	println("point clouds size mismatch");
end
d = zeros(Float32,length(subsP1)*3,npc);
for i=1:npc
	sigmaH = getDefaultHeavySide();
	
	u,I1,J1,V1 = ParamLevelSetModelFunc(Mesh,m[:,i];computeJacobian = 1,sigma = sigmaH,bf = 1,numParamOfRBF = numParamOfRBF);
	J1 = sparse(I1,J1,V1,prod(n),length(m[:,i]));
	
	
	
	nx = A1*(D1'*u);
	ny = A2*(D2'*u);
	nz = A3*(D3'*u);
	
	
	#########################Normalized version:
	#eps = 1e-15;
	#Af = getFaceAverageMatrix(Mesh);
	#Div = getDivergenceMatrix(Mesh);
	#writedlm(string("checknormals",".txt"),convert(Array{Float64},A1*(D1'*u)./sqrt.((A1*D1'*u).^2)));
	#nx = A1*(D1'*u) ./ sqrt.((A1*(D1'*u)).^2 .+ (A2*(D2'*u)).^2 .+ (A3*(D3'*u)).^2 .+ eps);  
	#ny = A2*(D2'*u) ./ sqrt.((A1*(D1'*u)).^2 .+ (A2*(D2'*u)).^2 .+ (A3*(D3'*u)).^2 .+ eps);  
	#nz = A3*(D3'*u)./ sqrt.((A1*(D1'*u)).^2 .+ (A2*(D2'*u)).^2 .+ (A3*(D3'*u)).^2 .+ eps);  
	#norms = [nx ny nz];
	#writedlm(string("curnormals",".txt"),convert(Array{Float64},norms));

	cursubs = subs[round(Int,(i-1)*(1/npc - margin)*length(subs))+1:min(length(subs),round(Int,i*(1/npc + margin)*length(subs)))];
	b = tuple(pFor.b[i,:]...);
	ind = subv2ind(pFor.Mesh.n,cursubs);
	I2 = collect(1:length(ind));
	J2 = ind;
	V2 = ones(size(ind));
	P = sparse(I2,J2,V2,length(ind),length(u));
	nxt = P*nx; nyt = P*ny; nzt = P*nz;
	curnormals = [nxt nyt nzt];
	d[:,i] = curnormals[:];
	
	dRx = P*A1*D1';
	dRy = P*A2*D2';
	dRz = P*A3*D3';
	
	Jacobians[i] = J1'*[dRx; dRy; dRz]';
	
end
JacT = blockdiag(Jacobians...);
return d,JacT
end
