import LinearAlgebra
using LinearAlgebra
using SparseArrays
using jInv
### matrix free S (slicing) matrix

function binfunc(ii::Int32,binFactor::Int32)
	ii -=1;
	ii = div(ii,binFactor);
	ii +=1;
	return ii;
end

export sampleSlices,sampleSlicesT
function sampleSlices(u::Array{Float64,3},cellVolume::Float64,samplingBinning::Int64=1)
n_tup = size(u);
n = collect(n_tup);
traceLength = div(n[3],samplingBinning);
s = zeros(eltype(u),traceLength);
for k = 1:traceLength
	@inbounds s[k] = sum(view(u,:,:,(samplingBinning*(k-1)+1):(samplingBinning*k)));
end
s.*=cellVolume;
return s;
end

function sampleSlicesT(s::Array,cellVolume::Float64,u::Array{Float64,3})
traceLength = length(s);
samplingBinning = div(size(u,3),traceLength);
for k = 1:traceLength
	@inbounds u[:,:,(samplingBinning*(k-1)+1):(samplingBinning*k)] = cellVolume*s[k];
end
end


export getData
function getData(m::Array,pFor::PointCloudParam,doClear::Bool=false)
d = 0;
Mesh = pFor.Mesh;
n = Mesh.n;
ndips = 1;
samplingBinning = pFor.samplingBinning;

if pFor.method == MATFree
	Div  = getDivergenceMatrix(Mesh)
	println("Div size:", size(Div));
	eps = 1e-4/Mesh.h[1];
	Af   = getFaceAverageMatrix(Mesh)
	A1 = Af[:,1:Int(size(Af,2)/3)]; println("A1 size:",size(A1))
	A2 = Af[:,Int(size(Af,2)/3)+1:2*Int(size(Af,2)/3)];
	A3 = Af[:,2*Int(size(Af,2)/3)+1:Int(size(Af,2))];
	
	
	D1 = kron(sparse(1.0I, Mesh.n[3], Mesh.n[3]),kron(sparse(1.0I, Mesh.n[2], Mesh.n[2]),ddx(Mesh.n[1])))
	D2 = kron(sparse(1.0I, Mesh.n[3], Mesh.n[3]),kron(ddx(Mesh.n[2]),sparse(1.0I, Mesh.n[1], Mesh.n[1])))
	D3 = kron(ddx(Mesh.n[3]),kron(sparse(1.0I, Mesh.n[2], Mesh.n[2]),sparse(1.0I, Mesh.n[1], Mesh.n[1])))
	

	ind = pFor.P;
	I2 = collect(1:length(ind));
	J2 = ind;
	V2 = ones(size(ind));
	P = sparse(I2,J2,V2,length(ind),length(m));

	nx = pFor.Normals[:,1]; ny = pFor.Normals[:,2]; nz = pFor.Normals[:,3];
	
	dRx = 2*(spdiagm( 0 => vec(1 ./ ((P*A1*(D1'*m).^2 .+eps)))))*((1.0I-spdiagm( 0 => vec(nx)))*P*A1*D1');
	dRy = 2*(spdiagm( 0 => vec(1 ./ ((P*A2*(D2'*m).^2 .+eps)))))*((1.0I-spdiagm( 0 => vec(ny)))*P*A2*D2');
	dRz = 2*(spdiagm( 0 => vec(1 ./ ((P*A3*(D3'*m).^2 .+eps)))))*((1.0I-spdiagm( 0 => vec(nz)))*P*A3*D3');
	
	pFor.Jacobian = [dRx; dRy; dRz];
	
	return pFor.Normals,pFor;
	end
	
if pFor.method == RBFBased || pFor.method == RBF10Based || pFor.method == RBF5Based
	if pFor.method == RBF10Based
		numParamOfRBF = 10;
	else
		numParamOfRBF = 5;
	end
	
	sigmaH = getDefaultHeavySide();
	u,I1,J1,V1 = ParamLevelSetModelFunc(Mesh,m;computeJacobian = 1,sigma = sigmaH,bf = 1,numParamOfRBF = numParamOfRBF);
	J1 = sparse(I1,J1,V1,prod(n),length(m));
	d = u;
	
	Div  = getDivergenceMatrix(Mesh)
	println("Div size:", size(Div));
	eps = 1e-4/Mesh.h[1];
	Af   = getFaceAverageMatrix(Mesh)
	A1 = Af[:,1:Int(size(Af,2)/3)]; println("A1 size:",size(A1))
	A2 = Af[:,Int(size(Af,2)/3)+1:2*Int(size(Af,2)/3)];
	A3 = Af[:,2*Int(size(Af,2)/3)+1:Int(size(Af,2))];
	
	
	D1 = kron(sparse(1.0I, Mesh.n[3], Mesh.n[3]),kron(sparse(1.0I, Mesh.n[2], Mesh.n[2]),ddx(Mesh.n[1])))
	D2 = kron(sparse(1.0I, Mesh.n[3], Mesh.n[3]),kron(ddx(Mesh.n[2]),sparse(1.0I, Mesh.n[1], Mesh.n[1])))
	D3 = kron(ddx(Mesh.n[3]),kron(sparse(1.0I, Mesh.n[2], Mesh.n[2]),sparse(1.0I, Mesh.n[1], Mesh.n[1])))
	
	
	#P is a sparse matrix of size k \times n, where k = #point on point cloud , n = volume size of the mesh
	indices = pFor.P;
	I2 = collect(1:length(indices));
	J2 = indices;
	V2 = ones(size(indices));
	#println("length indices:",length(indices),"length u:",length(u))
	P = sparse(I2,J2,V2,length(indices),length(u));
	

	nx = P*A1*(D1'*u);#./ (P*sqrt.((A1*D1'*u).^2 .+ eps));
	ny = P*A2*(D2'*u);#./ (P*sqrt.((A2*D2'*u).^2 .+ eps));
	nz = P*A3*(D3'*u);#./ (P*sqrt.((A3*D3'*u).^2 .+ eps));

	#d2R = 2*(spdiagm( 0 => vec(1 ./ ((P*Af*(Div'*u).^2 .+eps)))))*((1.0I-spdiagm( 0 => vec(Wf)))*P*Af*Div');
	
	dRx = P*A1*D1';
	dRy = P*A2*D2';
	dRz = P*A3*D3';
    d2R = [dRx; dRy; dRz];

	 
	Wf = [nx ny nz];
	d = ((Wf))
	pFor.Jacobian =  ((J1)'*d2R')';
end
return d,pFor;
end


function getDirectDataRBF(pFor, m,numParamOfRBF = 5)
samplingBinning = pFor.samplingBinning;
Mesh = pFor.Mesh;
n = pFor.Mesh.n;
h = pFor.Mesh.h;
traceLength = div(n[3],samplingBinning);
ndips = 1;
d = zeros(Float32,traceLength,ndips);
lengthRBFparams = size(m,1);
Ihuge = zeros(Int32,0);
I1 = zeros(Int32,0);
J1 = zeros(Int32,0);
V1 = zeros(Float64,0);
sigmaH = getDefaultHeavySide();
u = zeros(prod(n));
dsu = zeros(prod(n));
Xc = convert(Array{Float32,2},getCellCenteredGrid(Mesh));
binningFactor = convert(Int32,n[1]*n[2]*samplingBinning);
Jacobians = Array{SparseMatrixCSC{Float64,Int32}}(ndips);

iifunc = (ii)->binfunc(ii,binningFactor)
nz = 1;
volCell = prod(Mesh.h);
for ii = 1:ndips
	u = vec(u);
	u,I1,J1,V1,Ihuge = ParamLevelSetModelFunc(Mesh,m;computeJacobian = 1,sigma = sigmaH,
				Xc = Xc,u = u,dsu = dsu,Ihuge = Ihuge,Is = I1, Js = J1,Vs = V1,iifunc = iifunc,numParamOfRBF=numParamOfRBF);
	u = reshape(u,tuple(n...));
	d[:,ii] = sampleSlices(u,volCell,samplingBinning);
	Jacobians[ii] = sparse(J1,I1,V1,lengthRBFparams,traceLength);
	(Jacobians[ii].nzval).*=volCell;
end
JacT = blockdiag(Jacobians...);
# JacT = 0.0;

return d,JacT
end
