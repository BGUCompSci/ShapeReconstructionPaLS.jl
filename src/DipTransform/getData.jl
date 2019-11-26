
### matrix free S (slicing) matrix

function binfunc(ii::Int64,binFactor::Int64)
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
	@inbounds u[:,:,(samplingBinning*(k-1)+1):(samplingBinning*k)] .= cellVolume*s[k];
end
end


export getData
function getData(m::Array,pFor::DipParam,doClear::Bool=false)
d = 0;
Mesh = pFor.Mesh;
n = Mesh.n;
theta_phi = pFor.theta_phi_rad;
ndips = size(theta_phi,1);
samplingBinning = pFor.samplingBinning;

b = pFor.b;

if mod(n[3],samplingBinning)!=0
	error("n[3] must divide to sampling binning.");
end

if pFor.method == MATBased
	# println("MAT version")
	if norm(b) > 0.0
		warn("MATBased does not support built in translation.");
	end
	if length(pFor.Jacobian) == 0
		S = generateSamplingMatrix(Mesh,theta_phi,samplingBinning);
		pFor.Jacobian = S;
	end
	d = pFor.Jacobian'*m[:];
	d = reshape(d,div(length(d),ndips),ndips);
elseif pFor.method == MATFree
	# println("MAT-free version")

	traceLength = div(n[3],samplingBinning);
	d = zeros(Float32,traceLength,ndips);
	m = reshape(m,tuple(n...));
	mr = zeros(eltype(m),size(m));
	XT = zeros(0);
	XTT = zeros(0);
	doTranspose = false;
	for ii = 1:ndips
		(mr,XT,XTT) = rotateAndMove3D(m,theta_phi[ii,:],(b[ii,:]./Mesh.h),doTranspose,mr,XT,XTT);
		d[:,ii] = sampleSlices(mr,prod(Mesh.h),samplingBinning);
	end
 #-----------Compare MATBased and MATFree

	# if norm(b) > 0.0
	# 	warn("MATBased does not support built in translation.");
	# end
	# if length(pFor.Jacobian) == 0
	# 	S = generateSamplingMatrix(Mesh,theta_phi,samplingBinning);
	# 	pFor.Jacobian = S;
	# end
	# d2 = pFor.Jacobian'*m[:];
	# d2 = reshape(d2,div(length(d2),ndips),ndips);
	# if norm(d-d2) .> 1e-5
	# 	warn("Bug");
	# else
	# 	warn("ITS ALL GOOD")
	# end
	# nn=collect([48 48 48])
	# m = reshape(d,tuple(nn...));
	# mr = zeros(eltype(m),size(m));
	# #NOW COMPARE Rotation Transpose:
	# uu = zeros(Float32,traceLength,ndips);
	# for ii = 1:ndips
	# 	#(mr,XT,XTT) = rotateAndMove3D(m,theta_phi[ii,:],(b[ii,:]./Mesh.h),mr,XT,XTT);
	# 	tmpd = rotateAndMove3DTranspose(d,theta_phi[ii,:],(b[ii,:]./Mesh.h))
	# 	uu[:,ii] = sampleSlices(tmpd,prod(Mesh.h),1);
	# end
	# println(string("size of uu=",size(uu)));
	#
	# uu_mbased = pFor.Jacobian*(pFor.Jacobian'*m[:]);
	# println(string("size of uu_mbased=",size(uu_mbased)));
	#
	# uu_mbased = reshape(uu_mbased,div(length(uu_mbased),ndips),ndips);
	#
	# if norm(uu-uu_mbased) .> 1e-5
	# 	warn("Bug INVERSE TRANSFORM");
	# else
	# 	warn("INVERSE TRANSFORM IS GOOD")
	# end
	# println(norm(d-d2))

#-------------------------------------------------------------------


elseif pFor.method == RBFBasedSimple1 || pFor.method == RBF10BasedSimple1
	if pFor.method == RBF10BasedSimple1
		numParamOfRBF = 10;
	else
		numParamOfRBF = 5;
	end
	if norm(b) > 0.0
		warn("RBFBasedSimple1 and RBF10BasedSimple1 do not support built in translation.");
	end
	sigmaH = getDefaultHeaviside();
	u,JBuilder = ParamLevelSetModelFunc(Mesh,m;computeJacobian = 1,sigma = sigmaH,bf = 1,numParamOfRBF = numParamOfRBF);
	# J1 = sparse(I1,J1,V1,prod(n),length(m));
	J1 = getSparseMatrix(JBuilder);
	if length(pFor.S)==0
		pFor.S = generateSamplingMatrix(Mesh,theta_phi,samplingBinning);
	end
	d = pFor.S'*u[:];
	d = reshape(d,div(length(d),ndips),ndips);
	pFor.Jacobian = J1'*pFor.S;
elseif pFor.method == RBFBasedSimple2 || pFor.method == RBF10BasedSimple2
	if pFor.method == RBF10BasedSimple2
		numParamOfRBF = 10;
	else
		numParamOfRBF = 5;
	end
	mrot,Jrot = rotateAndMoveRBFsimple(m,Mesh,theta_phi,b;computeJacobian = 1,numParamOfRBF=numParamOfRBF);
	d,JacT = getDipDataRBF(pFor, mrot,theta_phi,b,numParamOfRBF);
	# multiply Jacobian with Jrot, and then take the transpose.
	pFor.Jacobian = Jrot'*JacT;
elseif pFor.method == RBFBased || pFor.method == RBF10Based
	if pFor.method == RBF10Based
		numParamOfRBF = 10;
	else
		numParamOfRBF = 5;
	end
	## here theta_phi is not constant - so we ignore the theta_phi in pFor
	nRBF 				= div(length(m)- 5*pFor.ndipsAll,numParamOfRBF) ;
	(m1,theta_phi,b) 	= splitRBFparamAndRotationsTranslations(m,nRBF,pFor.ndipsAll,numParamOfRBF);
	theta_phi 			= theta_phi[pFor.workerSubIdxs,:];
	b 					= b[pFor.workerSubIdxs,:];
	mrot,Jrot 			= rotateAndMoveRBF(m1,Mesh,theta_phi,b;computeJacobian = 1,numParamOfRBF=numParamOfRBF);
	d,JacT 				= getDipDataRBF(pFor, mrot,theta_phi,b,numParamOfRBF);
	# multiply Jacobian with Jrot, and then take the transpose.

	JacobianT = convert(SparseMatrixCSC{Float64,Int32},spzeros(length(m),prod(size(d))));
	IpIdxs = getIpIdxs(pFor.workerSubIdxs,nRBF,pFor.ndipsAll,numParamOfRBF);

	JacobianT[IpIdxs,:] = Jrot'*JacT;
	pFor.Jacobian = JacobianT;
	# pFor.Jacobian 		= Jrot'*JacT;
end
return d,pFor;
end


function getDipDataRBF(pFor, mrot,theta_phi,b,numParamOfRBF = 5)
samplingBinning = pFor.samplingBinning;
Mesh = pFor.Mesh;
n = pFor.Mesh.n;
h = pFor.Mesh.h;
traceLength = div(n[3],samplingBinning);
ndips = size(theta_phi,1);
d = zeros(Float32,traceLength,ndips);
lengthRBFparams = size(mrot,1);
JBuilder = getSpMatBuilder(Int64,Float64,traceLength,lengthRBFparams,10*traceLength)
sigmaH = getDefaultHeaviside();
u = zeros(prod(n));
dsu = zeros(prod(n));
Xc = convert(Array{Float32,2},getCellCenteredGrid(Mesh));
binningFactor = n[1]*n[2]*samplingBinning;
Jacobians = Array{SparseMatrixCSC{Float64,Int64}}(undef, ndips);

iifunc = (ii)->binfunc(ii,binningFactor)
nz = 1;
volCell = prod(Mesh.h);
for ii = 1:ndips
	u = vec(u);
	u,JBuilder = ParamLevelSetModelFunc(Mesh,mrot[:,ii];computeJacobian = 1,sigma = sigmaH,
				Xc = Xc,u = u,dsu = dsu,Jbuilder = JBuilder,iifunc = iifunc,numParamOfRBF=numParamOfRBF);
	u = reshape(u,tuple(n...));
	d[:,ii] = sampleSlices(u,volCell,samplingBinning);
	Jacobians[ii] = getSparseMatrixTransposed(JBuilder);
	(Jacobians[ii].nzval).*=volCell;
end
JacT = blockdiag(Jacobians...);
# JacT = 0.0;
return d,JacT
end
