
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
function getData(m::Array,pFor::DirectParam,doClear::Bool=false)
d = 0;
Mesh = pFor.Mesh;
n = Mesh.n;
ndips = 1;
samplingBinning = pFor.samplingBinning;

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
	#d = reshape(d,div(length(d),ndips),ndips);
	println("getData jacobian:",size(J1))
	#println("Jrot size:",size(Jrot))
	pFor.Jacobian = J1;
# elseif pFor.method == RBFBasedSimple2 || pFor.method == RBF10BasedSimple2
	# if pFor.method == RBF10BasedSimple2
		# numParamOfRBF = 10;
	# else
		# numParamOfRBF = 5;
	# end
	# #mrot,Jrot = rotateAndMoveRBFsimple(m,Mesh,theta_phi,b;computeJacobian = 1,numParamOfRBF=numParamOfRBF);
	# d,JacT = getDipDataRBF(pFor, m,numParamOfRBF);
	# # multiply Jacobian with Jrot, and then take the transpose.
	# pFor.Jacobian = JacT;
# elseif pFor.method == RBFBased || pFor.method == RBF10Based
	# if pFor.method == RBF10Based
		# numParamOfRBF = 10;
	# else
		# numParamOfRBF = 5;
	# end
	# ## here theta_phi is not constant - so we ignore the theta_phi in pFor
	# nRBF 				= div(length(m)- 5*pFor.ndipsAll,numParamOfRBF) ;
	# (m1,theta_phi,b) 	= splitRBFparamAndRotationsTranslations(m,nRBF,pFor.ndipsAll,numParamOfRBF);
	# theta_phi 			= theta_phi[pFor.workerSubIdxs,:];
	# b 					= b[pFor.workerSubIdxs,:];
	# mrot,Jrot 			= rotateAndMoveRBF(m1,Mesh,theta_phi,b;computeJacobian = 1,numParamOfRBF=numParamOfRBF);
	# d,JacT 				= getDipDataRBF(pFor, mrot,theta_phi,b,numParamOfRBF);
	# # multiply Jacobian with Jrot, and then take the transpose.

	# JacobianT = convert(SparseMatrixCSC{Float64,Int32},spzeros(length(m),prod(size(d))));
	# IpIdxs = getIpIdxs(pFor.workerSubIdxs,nRBF,pFor.ndipsAll,numParamOfRBF);

	# JacobianT[IpIdxs,:] = Jrot'*JacT;
	# pFor.Jacobian = JacobianT;
	# # pFor.Jacobian 		= Jrot'*JacT;
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
JacT = blkdiag(Jacobians...);
# JacT = 0.0;

return d,JacT
end
