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
#	println("len of v:",size(v))
#	println("len of m:",size(m))
	S = pFor.Jacobian;
#	println("len of J:",size(S))
	JtV = S'*v[:];
return JtV;
end
