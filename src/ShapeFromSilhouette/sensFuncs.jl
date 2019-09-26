import jInv.ForwardShare.getSensMat
function getSensMat(m::Vector,pFor::SfSParam)
	if pFor.method == MATFree || pFor.method == MATBased
		error("getSensMat: Not supported or too expensive");
	end
	return pFor.Jacobian';
end


export getSensMatVec
function getSensMatVec(v::Vector,m::Vector,pFor::SfSParam)
if pFor.method == MATFree
	nshots = size(pFor.theta_phi_rad,1);
	traceLength = prod(pFor.ScreenMeshX2X3.n);
	n = pFor.ObjMesh.n;
	d = zeros(Float32,traceLength,nshots);
	v = reshape(v,tuple(n...));
	vr = zeros(eltype(v),size(v));
	XT = zeros(0);
	XTT = zeros(0);
	b = zeros(nshots,3);
	for ii = 1:nshots
		(vr,XT,XTT) = rotateAndMove3D(v,pFor.theta_phi_rad[ii,:],b[ii,:],vr,XT,XTT);
		# d[:,ii] = pFor.SampleMatT'*vr[:];
		d[:,ii] = softMaxSensMatVec(pFor.SampleMatT,m,vr[:]);
	end
	# d = d[:].*pFor.Jacobian;
	# d = reshape(d,traceLength,nshots);
	return d;
else
	S = pFor.Jacobian;
	d = S'*v[:];
	nshots = size(pFor.theta_phi_rad,1);
	d = reshape(d,div(length(d),nshots),nshots);
end
end
export getSensTMatVec
function getSensTMatVec(v::Vector,m::Vector,pFor::SfSParam)
JtV = 0;	
if pFor.method == MATFree
	error("Implement this!!!");
	Mesh = pFor.Mesh;
	n = Mesh.n;
	theta_phi = pFor.theta_phi_rad;
	ndips = size(theta_phi,1);
	sampleBinning = pFor.samplingBinning;
	traceLength = div(n[3],sampleBinning);
	d = reshape(v,traceLength,ndips);
	u = zeros(tuple(n...));
	JtV = zeros(size(m));
	for ii = 1:ndips
		sampleSlicesT(d[:,ii],prod(Mesh.h),u);
		JtV += rotateAndMove3DTranspose(u,theta_phi[ii,:],zeros(3))[:];
	end
else
	S = pFor.Jacobian;
	JtV = S*v[:];
end
return JtV;
end
