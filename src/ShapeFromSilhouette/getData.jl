using SparseArrays

SigmoidFunc(x) =(0.5*tanh.(5*(x-0.5))+0.5 + (1-0.5*tanh.(5*(1-0.5))-0.5).*2 .*(x-0.5));
dsigmoidFunc(x) = 0.0133857 + 2.5*((sech.(5*(-0.5 + x))).^2);
export getData
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

heavySideVisData = (d)->heavySide!(d,copy(d),0.5,0.5); 

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
	# println("MAT-free version")
	
	
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
		(mr,XT,XTT) = rotateAndMove3D(m,theta_phi[ii,:],b[ii,:],doTranspose,mr,XT,XTT);
		# d[:,ii] = pFor.SampleMatT'*mr[:];
		d[:,ii] = softMaxProjection(pFor.SampleMatT,mr[:]);
	end
	# if isempty(mask)
		# dsd = heavySideVisData(d);
		# pFor.Jacobian = dsd[:]; ## this is a bit of a hack...
	# else
		# dt = copy(d);
		# dsd = heavySideVisData(d);
		# d = mask.*d + (1.0-mask).*dt;
		# pFor.Jacobian = mask.*(dsd[:]) + (1.0-mask); ## this is a bit of a hack...
	# end
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
	d,JacT = getVisDataRBF(pFor, mrot,theta_phi,b,mask,heavySideVisData,numParamOfRBF);
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
	d,JacT 				= getVisDataRBF(pFor, mrot,theta_phi,b,mask,heavySideVisData,numParamOfRBF);
	# multiply Jacobian with Jrot, and then take the transpose.
	Idxs = 				getIpIdxs
	
	JacobianT = convert(SparseMatrixCSC{Float64,Int32},spzeros(length(m),prod(size(d))));
	IpIdxs = getIpIdxs(pFor.workerSubIdxs,nRBF,pFor.nshotsAll,numParamOfRBF);
	#mrot,Jrot = rotateAndMoveRBF(m1,Mesh,theta_phi,b;computeJacobian = 1,numParamOfRBF=numParamOfRBF);
	
	JacobianT[IpIdxs,:] = Jrot'*JacT;
	pFor.Jacobian = JacobianT; 
	
	# pFor.Jacobian 		= Jrot'*JacT;
end
return d,pFor;
end


function getVisDataRBF(pFor, mrot,theta_phi,b,mask,heavySideVisData::Function,numParamOfRBF = 5)
Mesh = pFor.ObjMesh;
n = Mesh.n;
h = Mesh.h;
traceLength = prod(pFor.ScreenMeshX2X3.n);
nshots = size(theta_phi,1);
d = zeros(Float32,traceLength,nshots);
lengthRBFparams = size(mrot,1);
Ihuge = zeros(Int32,0);
I1 = zeros(Int32,0);
J1 = zeros(Int32,0);
V1 = zeros(Float64,0);
sigmaH = getDefaultHeavySide();
u = zeros(prod(n));
dsu = zeros(prod(n));
Xc = convert(Array{Float32,2},getCellCenteredGrid(Mesh));

Jacobians = Array{SparseMatrixCSC{Float64,Int32}}(undef,nshots);

nz = 1;
volCell = prod(Mesh.h);

for ii = 1:nshots
	u,I1,J1,V1,Ihuge = ParamLevelSetModelFunc(Mesh,mrot[:,ii];computeJacobian = 1,sigma = sigmaH,
										Xc = Xc,u = u,dsu = dsu,Ihuge = Ihuge,Is = I1, Js = J1,Vs = V1,numParamOfRBF=numParamOfRBF);
	
	
	#d[:,ii],Jii = softMaxProjWithSensMat(pFor.SampleMatT,u);
	
	d[:,ii],Jii = softMaxProjWithSigmoid(pFor.SampleMatT,u)
	
	Jii = (sparse(J1,I1,V1,lengthRBFparams,prod(n))*Jii);
	# dii = pFor.SampleMatT'*u[:];
	# Jii = sparse(J1,I1,V1,lengthRBFparams,prod(n))*pFor.SampleMatT;
	# if isempty(mask)
		# dsdii = heavySideVisData(dii);
		# Jii = Jii*spdiagm(dsdii[:]);
	# else
		# maskii = mask[:,ii];
		# dt = copy(dii);
		# dsdii = heavySideVisData(dii);
		# dii = maskii.*dii + (1.0-maskii).*dt;
		# Jii = Jii*spdiagm(maskii.*(dsdii[:]) + (1.0-maskii));
	# end 
	# d[:,ii] = dii;
	Jacobians[ii] = Jii;
end
JacT = blkdiag(Jacobians...);
# JacT = 0.0;
return d,JacT
end


export softMaxProjection
function softMaxProjection(Proj::SparseMatrixCSC,u::Vector{Float64})
	etta = 50.0;
	y  = exp.(etta*u);
	Ay = Proj'*y;
	v  = y.*u;
	Av = Proj'*v;
	ans = Av./Ay;
	return ans;
end


export softMaxSensMatVec
function softMaxSensMatVec(Proj::SparseMatrixCSC,u::Vector{Float64},vec::Vector{Float64})
	etta = 50.0;
	y  = exp.(etta*u);
	Ay = Proj'*y;
	v  = y.*u;
	Av = Proj'*v;
	dy = etta*y;
	dv = etta*y.*u + y;
	AYinv = 1.0./(Ay + 1e-10);
	# Jt = spdiagm(dv)*Proj*spdiagm(AYinv) - spdiagm(dy)*Proj*spdiagm(AYinv.*AYinv.*Av);
	Jvec = AYinv.*(Proj'*(dv.*vec)) - AYinv.*AYinv.*Av.*(Proj'*(dy.*vec))
	return Jvec;
end


export softMaxProjWithSigmoid
function softMaxProjWithSigmoid(Proj::SparseMatrixCSC,u::Vector{Float64})
	etta = 50.0;
	
	Rays = u.*Proj;
	#ans=zeros(size(Rays,2));
	#Iterate over the rays:
	d  = diff(Rays);
	indices = find(d.<0);
	indices = indices .+ 1 ;

	#Proj[deleteat!(collect(1:length(Proj)),indices)] = 0;
	Proj[indices] .= 0;
	
	
	# for rayIdx = 1:size(Rays,2)
		# currRay = copy(Rays[:,rayIdx]);
		# d  = diff(currRay);
		# indices = find(d.<0);
		# # nonZind = findnz(Proj[:,rayIdx])
		# # nonZind = nonZind[1]
		# # #Find indices where we go up:
		# # indices=[];
		# # count=1;
		 # # for i=1:length(nonZind)-1#(length(currRay)-1)
			 # # if(count > 10)
				# # break;
			# # end
			# # iidx = nonZind[i]
			# # nextiidx = nonZind[i+1]
			# # if(currRay[iidx]<=currRay[nextiidx] && currRay[iidx]>0.5)
				# # append!(indices,iidx);
				# # count = count + 1;
			# # #elseif (count>1 && indices[count-1] == i-1) #that's the point of change (value decreases after this point)
			# # #	append!(indices,i);
			# # #	count = count + 1;
			# # end
		# # end
		# #currRay = currRay[indices];
		# #y = exp.(etta*currRay);
		# #softMax_ray = y./sum(y);
		# #ans[rayIdx] = SigmoidFunc(sum(softMax_ray.*currRay));
		
		# #Preparation for Jacobian computation:
		# currRay =(Proj[:,rayIdx]);
		# currRay[deleteat!(collect(1:length(currRay)),indices)] = 0;
		# Proj[:,rayIdx] = currRay;
	# end
	
	dropzeros(Proj)
	y  = exp.(etta*u);
	Ay = Proj'*y;
	v  = y.*u;
	Av = Proj'*v;
	dy = etta*y;
	dv = etta*y.*u + y;
	AYinv = 1.0./(Ay + 1e-10);
	ans2 = SigmoidFunc(Av.*AYinv);
	
	#Test for answer:
	#println("Diff in ans:",abs(ans2-ans))
	
	dsig = dsigmoidFunc(Av.*AYinv)
	
	Jtt = copy(Proj);
	
	dropzeros(Jtt)
	#Ans = sig(Av./Ay) => J_ans = dsig(Av./Ay)*(dAv./Ay) - dsig(Av./Ay)*(Av*dy/Ay*Ay) 
	# dsig(Av./Ay)*Proj*spdiagm(dv)*Proj*spdiagm(AYinv) - dsig(Av./Ay)
	# Jt = spdiagm(dsig(Av./Ay))*spdiagm(dv)*Proj*spdiagm(AYinv) - spdiagm(dsig(Av./Ay))*spdiagm(dy)*Proj*spdiagm(AYinv.*AYinv.*Av)
	for j = 1:size(Jtt,2)
		@inbounds AYinv_j = AYinv[j];
		@inbounds Av_j = Av[j];
		for gidx = Jtt.colptr[j]:(Jtt.colptr[j+1]-1)
			@inbounds nzv = Jtt.nzval[gidx];
			@inbounds i = Jtt.rowval[gidx];
			@inbounds Jtt.nzval[gidx] = dsig[j]*dv[i]*nzv*AYinv_j - dsig[j]*dy[i]*nzv*AYinv_j*AYinv_j*Av_j;
		end
	end
	dropzeros(Jtt)
	
	
	return ans2,Jtt;
end


export softMaxProjWithSensMat
function softMaxProjWithSensMat(Proj::SparseMatrixCSC,u::Vector{Float64})
	etta = 50.0;
	y  = exp.(etta*u);
	Ay = Proj'*y;
	v  = y.*u;
	Av = Proj'*v;
	ans = Av./Ay;
	
	dy = etta*y;
	dv = etta*y.*u + y;
	AYinv = 1.0./(Ay + 1e-10);
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
