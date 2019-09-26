import jInv.ForwardShare.getSensMat
function getSensMat(m::Vector,pFor::DipParam)
	if pFor.method == MATFree || pFor.method == MATBased
		error("getSensMat: Not supported or too expensive");
	end
	return pFor.Jacobian';
end


export getSensMatVec
function getSensMatVec(v::Vector,m::Vector,pFor::DipParam)
if pFor.method == MATFree
	return getData(v,pFor)[1];
else
	S = pFor.Jacobian;
	d = S'*v[:];
	ndips = size(pFor.theta_phi_rad,1);
	d = reshape(d,div(length(d),ndips),ndips);
end
end

export getSensTMatVec
function getSensTMatVec(v::Vector,m::Vector,pFor::DipParam)
JtV = 0;
if pFor.method == MATFree
	Mesh = pFor.Mesh;
	n = Mesh.n;
	theta_phi = pFor.theta_phi_rad;
	b = pFor.b;
	ndips = size(theta_phi,1);
	sampleBinning = pFor.samplingBinning;
	traceLength = div(n[3],sampleBinning);
	d = reshape(v,traceLength,ndips);
	u = zeros(tuple(n...));
	JtV = zeros(size(m));
	doTranspose = true;
	for ii = 1:ndips
		sampleSlicesT(d[:,ii],prod(Mesh.h),u);
		#JtV .+= rotateAndMove3DTranspose(u,theta_phi[ii,:],(b[ii,:]./Mesh.h))[:];
		(mr,XT,XTT) = rotateAndMove3D(u,theta_phi[ii,:],(b[ii,:]./Mesh.h),doTranspose);
		JtV .+= mr[:];
	end
else
	S = pFor.Jacobian;
	JtV = S*v[:];
end
return JtV;
end
