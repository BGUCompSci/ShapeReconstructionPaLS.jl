import jInv.ForwardShare.getSensMat
function getSensMat(m::Vector,pFor::PointCloudParam)
	return pFor.Jacobian;
end


export getSensMatVec
function getSensMatVec(v::Vector,m::Vector,pFor::PointCloudParam)
	println("Im in getSensMatVec")
	S = pFor.Jacobian;
	
	d = S'*v[:];

end

export getSensTMatVec
function getSensTMatVec(v::Vector,m::Vector,pFor::PointCloudParam)
	JtV = 0;
	S = pFor.Jacobian;
	JtV = S'*v[:];
return JtV;
end
