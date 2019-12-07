using SparseArrays
using LinearAlgebra
SigmoidFunc(x) =(0.5*tanh.(5*(x.-0.5)).+0.5 .+ (1-0.5*tanh.(5*(1-0.5))-0.5).*2 .*(x.-0.5));
dsigmoidFunc(x) = 0.0133857 .+ 2.5*((sech.(5*(-0.5 .+ x))).^2);
export getData,SigmoidFunc
function getData(m::Array,pFor::SfSParam,doClear::Bool=false)
d = 0;
Mesh = pFor.ObjMesh;
cellVolume = prod(Mesh.h);
mask = pFor.mask;
n = Mesh.n;
theta_phi = pFor.theta_phi_rad;
nshots = size(theta_phi,1);

numParamOfRBF = 5;

b = pFor.b;

heavisideVisData = (d)->heavySide!(d,copy(d),0.5,0.5); 

if pFor.method == MATBased
	error("Not working well. need to fix projection matrix.");
	# println("MAT version")
	# if length(pFor.Jacobian) == 0
		# S = generateProjectionOperatorWithRotation(pFor);
		# pFor.Jacobian = S;
	# end
	# d = pFor.Jacobian'*m[:];
	# d = reshape(d,div(length(d),nshots),nshots);
elseif pFor.method == MATFree
	if length(pFor.SampleMatT) == 0
		pFor.SampleMatT = generateProjectionOperator(pFor);
	end
	traceLength = prod(pFor.ScreenMeshX2X3.n);
	d = zeros(Float32,traceLength,nshots);
	m = reshape(m,tuple(n...));
	mr = zeros(eltype(m),size(m));
	XT = zeros(0);
	XTT = zeros(0);
	for ii = 1:nshots
		doTranspose = false;
		(mr,XT,XTT) = rotateAndMove3D(m,theta_phi[ii,:],(b[ii,:]./Mesh.h),doTranspose,mr,XT,XTT);
		d[:,ii] = softMaxProjection(pFor.SampleMatT,mr[:]);
	end
	
elseif pFor.method == RBFBasedSimple1 || pFor.method == RBF10BasedSimple1
	error("Not working well. need to fix projection matrix.");
	# if pFor.method == RBF10BasedSimple1
		# numParamOfRBF = 10;
	# else
		# numParamOfRBF = 5;
	# end
	# sigmaH = getDefaultHeavySide();
	# u,I1,J1,V1 = ParamLevelSetModelFunc(Mesh,m;computeJacobian = 1,sigma = sigmaH,bf = 1,numParamOfRBF = numParamOfRBF);
	# J1 = sparse(I1,J1,V1,prod(n),length(m));
	# if length(pFor.SampleMatT) == 0
		# pFor.SampleMatT = generateProjectionOperatorWithRotation(pFor);
	# end
	# d = pFor.SampleMatT'*u[:];
	# d = reshape(d,div(length(d),nshots),nshots);
	# pFor.Jacobian = J1'*pFor.SampleMatT;
elseif pFor.method == RBFBasedSimple2 || pFor.method == RBF10BasedSimple2
	if pFor.method == RBF10BasedSimple2
		numParamOfRBF = 10;
	else
		numParamOfRBF = 5;
	end
	if length(pFor.SampleMatT) == 0
		pFor.SampleMatT = generateProjectionOperator(pFor);
	end
	mrot,Jrot = rotateAndMoveRBFsimple(m,Mesh,theta_phi,b;computeJacobian = 1,numParamOfRBF=numParamOfRBF);
	d,JacT = getVisDataRBF(pFor, mrot,theta_phi,b,mask,heavisideVisData,numParamOfRBF);
	# multiply Jacobian with Jrot, and then take the transpose.
	pFor.Jacobian = Jrot'*JacT;
elseif pFor.method == RBFBased || pFor.method == RBF10Based
	if pFor.method == RBF10Based
		numParamOfRBF = 10;
	else
		numParamOfRBF = 5;
	end
	if length(pFor.SampleMatT) == 0
		pFor.SampleMatT = generateProjectionOperator(pFor);
	end
	## here theta_phi is not constant - so we ignore the theta_phi in pFor
	nRBF 				= div(length(m)- 5*pFor.nshotsAll,numParamOfRBF);
	(m1,theta_phi,b) 	= splitRBFparamAndRotationsTranslations(m,nRBF,pFor.nshotsAll,numParamOfRBF);
	theta_phi 			= theta_phi[pFor.workerSubIdxs,:];
	b 					= b[pFor.workerSubIdxs,:];
	mrot,Jrot 			= rotateAndMoveRBF(m1,Mesh,theta_phi,b;computeJacobian = 1,numParamOfRBF=numParamOfRBF);
	d,JacT 				= getVisDataRBF(pFor, mrot,theta_phi,b,mask,heavisideVisData,numParamOfRBF);
	# multiply Jacobian with Jrot, and then take the transpose.
	Idxs = 				getIpIdxs;
	
	JacobianT = convert(SparseMatrixCSC{Float64,Int32},spzeros(length(m),prod(size(d))));
	IpIdxs = getIpIdxs(pFor.workerSubIdxs,nRBF,pFor.nshotsAll,numParamOfRBF);	
	JacobianT[IpIdxs,:] = Jrot'*JacT;
	pFor.Jacobian = JacobianT; 
	GC.gc();
	# pFor.Jacobian 		= Jrot'*JacT;
end
return d,pFor;
end


function getVisDataRBF(pFor, mrot,theta_phi,b,mask,heavisideVisData::Function,numParamOfRBF = 5)
Mesh = pFor.ObjMesh;
n = Mesh.n;
h = Mesh.h;
traceLength = prod(pFor.ScreenMeshX2X3.n);
nshots = size(theta_phi,1);
d = zeros(Float32,traceLength,nshots);
lengthRBFparams = size(mrot,1);
JBuilder = getSpMatBuilder(Int64,Float64,prod(n),lengthRBFparams,10*prod(n))

sigmaH = getDefaultHeaviside();
u = zeros(prod(n));
dsu = zeros(prod(n));
Xc = convert(Array{Float32,2},getCellCenteredGrid(Mesh));

Jacobians = Array{SparseMatrixCSC{Float64,Int64}}(undef,nshots);

for ii = 1:nshots
	u,JBuilder = ParamLevelSetModelFunc(Mesh,mrot[:,ii];computeJacobian = 1,sigma = sigmaH,
										Xc = Xc,u = u,dsu = dsu,Jbuilder = JBuilder,numParamOfRBF=numParamOfRBF);
	
	
	# d[:,ii] = softMaxProjWithSensMat(pFor.SampleMatT,u);
	d[:,ii],Jii = softMaxProjWithSigmoid(pFor.SampleMatT,u)
	Jii = getSparseMatrixTransposed(JBuilder)*Jii;
	Jacobians[ii] = Jii;
end
JacT = blockdiag(Jacobians...);
return d,JacT
end

export softMaxProjWithSigmoid
function softMaxProjWithSigmoid(Proj_g::SparseMatrixCSC,u::Vector{Float64})
	etta = 40.0;
	Proj = u.*Proj_g;
	dropzeros!(Proj);
	for rayIdx = 1:size(Proj,2)
		# if Proj.colptr[rayIdx+1] > Proj.colptr[rayIdx]
			# println(Proj.nzval[Proj.colptr[rayIdx]:Proj.colptr[rayIdx+1]-1])
		# end
		kmax = Proj.colptr[rayIdx];
		for k = (Proj.colptr[rayIdx]+1):(Proj.colptr[rayIdx+1]-1)		
			if Proj.nzval[k] < Proj.nzval[kmax] + 1e-5
				@inbounds Proj.nzval[k] = 0.0;
			else
				kmax = k;
			end
		end
		# if Proj.colptr[rayIdx+1] > Proj.colptr[rayIdx]
			# println(Proj.nzval[Proj.colptr[rayIdx]:Proj.colptr[rayIdx+1]-1])
			# println("~~~~~~~~~~~~~~~~~~~")
		# end
		
	end
	dropzeros!(Proj)
	Proj.nzval[:].=1.0;
	relevant = findall(sum(Proj,dims=2)[:].> 1e-16);
	y = zeros(size(u));
	v = zeros(size(u));
	y[relevant]  = exp.(etta*u[relevant]);
	Ay = Proj'*y;
	v[relevant]  = y[relevant].*u[relevant];
	Av = Proj'*v;
	AYinv = 1.0./(Ay .+ 1e-7);
	ans2 = SigmoidFunc(Av.*AYinv);
	dsig = dsigmoidFunc(Av.*AYinv)
	
	dy = etta*y;
	dv = etta*y.*u + y;
	
	
	#Ans = sig(Av./Ay) => J_ans = dsig(Av./Ay)*(dAv./Ay) - dsig(Av./Ay)*(Av*dy/Ay*Ay) 
	# dsig(Av./Ay)*Proj*spdiagm(dv)*Proj*spdiagm(AYinv) - dsig(Av./Ay)
	# Jt = spdiagm(dsig(Av./Ay))*spdiagm(dv)*Proj*spdiagm(AYinv) - spdiagm(dsig(Av./Ay))*spdiagm(dy)*Proj*spdiagm(AYinv.*AYinv.*Av)
	for j = 1:size(Proj,2)
		@inbounds AYinv_j = AYinv[j];
		@inbounds Av_j = Av[j];
		for gidx = Proj.colptr[j]:(Proj.colptr[j+1]-1)
			@inbounds nzv = Proj.nzval[gidx];
			@inbounds i = Proj.rowval[gidx];
			@inbounds Proj.nzval[gidx] = dsig[j]*dv[i]*nzv*AYinv_j - dsig[j]*dy[i]*nzv*AYinv_j*AYinv_j*Av_j;
		end
	end
	dropzeros(Proj);
	return ans2,Proj;
end

export softMaxProjection
function softMaxProjection(Proj::SparseMatrixCSC,u::Vector{Float64})
	etta = 40.0;
	y  = exp.(etta*u);
	Ay = Proj'*y;
	v  = y.*u;
	Av = Proj'*v;
	ans = Av./Ay;
	return ans;
end


export softMaxSensMatVec
function softMaxSensMatVec(Proj::SparseMatrixCSC,u::Vector{Float64},vec::Vector{Float64})
	etta = 40.0;
	y  = exp.(etta*u);
	Ay = Proj'*y;
	v  = y.*u;
	Av = Proj'*v;
	dy = etta*y;
	dv = etta*y.*u + y;
	AYinv = 1.0./(Ay .+ 1e-7);
	# Jt = spdiagm(dv)*Proj*spdiagm(AYinv) - spdiagm(dy)*Proj*spdiagm(AYinv.*AYinv.*Av);
	Jvec = AYinv.*(Proj'*(dv.*vec)) - AYinv.*AYinv.*Av.*(Proj'*(dy.*vec))
	return Jvec;
end
export softMaxProjWithSensMat
function softMaxProjWithSensMat(Proj::SparseMatrixCSC,u::Vector{Float64})
	etta = 40.0;
	y  = exp.(etta*u);
	Ay = Proj'*y;
	v  = y.*u;
	Av = Proj'*v;
	ans = Av./Ay;
	
	dy = etta*y;
	dv = etta*y.*u + y;
	AYinv = 1.0./(Ay .+ 1e-7);
	# The code below does this line only much faster:
	# Jt = spdiagm(dv)*Proj*spdiagm(AYinv) - spdiagm(dy)*Proj*spdiagm(AYinv.*AYinv.*Av);
	Jtt = copy(Proj);
	for j = 1:size(Jtt,2)
		@inbounds AYinv_j = AYinv[j];
		@inbounds Av_j = Av[j];
		for gidx = Jtt.colptr[j]:(Jtt.colptr[j+1]-1)
			@inbounds nzv = Jtt.nzval[gidx];
			@inbounds i = Jtt.rowval[gidx];
			@inbounds Jtt.nzval[gidx] = dv[i]*nzv*AYinv_j - dy[i]*nzv*AYinv_j*AYinv_j*Av_j;
		end
	end
	# println(norm(Jt - Jtt,1));
	
	return ans,Jtt;
end
