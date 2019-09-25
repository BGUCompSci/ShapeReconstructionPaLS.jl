

function loc2cs3D(loc1::Union{Int64,Array{Int64}},loc2::Union{Int64,Array{Int64}},loc3::Union{Int64,Array{Int64}},n::Array{Int64,1})
@inbounds cs = loc1 .+ (loc2.-1)*n[1] .+ (loc3.-1)*n[1]*n[2];
return cs;
end


# function getDippingMatrix(Mesh::RegularMesh)
# S = kron(speye(Mesh.n[3]),spones(1,Mesh.n[1]*Mesh.n[2]));
# return S;
# end
export generateSamplingMatrix
function generateSamplingMatrix(Mesh::RegularMesh,theta_phi_rad::Array{Float64,2},samplingBinning::Int64 = 1,calcSR = 1)
X = getCellCenteredGrid(Mesh);

mid_domain = [(Mesh.domain[1]+Mesh.domain[2])/2,(Mesh.domain[3]+Mesh.domain[4])/2,(Mesh.domain[5]+Mesh.domain[6])/2];
half_length_domain = [(Mesh.domain[2]-Mesh.domain[1])/2,(Mesh.domain[4]-Mesh.domain[3])/2,(Mesh.domain[6]-Mesh.domain[5])/2];
X[:,1] .= X[:,1] .- mid_domain[1];
X[:,2] .= X[:,2] .- mid_domain[2];
X[:,3] .= X[:,3] .- mid_domain[3];

num_rotations = size(theta_phi_rad,1);
offset = 0;
n =  Mesh.n;
trace_length = Mesh.n[1]*Mesh.n[2];
volume = prod(Mesh.n);

globalRowIdx = zeros(Int32,(volume*num_rotations+5));
globalColIdx = zeros(Int32,(volume*num_rotations+5));


R = 0;
offset = 0;
# println("Begin generating matrix");
# tic()
for k=1:num_rotations
	Rk = inv(getRotate3D(theta_phi_rad[k,1],theta_phi_rad[k,2]));
	Xk = X*Rk';
	# Here we move the domain to be [0,L] to compute the coordinates.
	Xk[:,1] .= (Xk[:,1] .+ half_length_domain[1])*(1.0 ./Mesh.h[1]) .+ 0.5;
	Xk[:,2] .= (Xk[:,2] .+ half_length_domain[2])*(1.0 ./Mesh.h[2]) .+ 0.5;
	Xk[:,3] .= (Xk[:,3] .+ half_length_domain[3])*(1.0 ./Mesh.h[3]) .+ 0.5;
	Xk = round.(Int64,Xk);
	II = findall((Xk[:,1].>=1) .& (Xk[:,1].<=Mesh.n[1]) .& (Xk[:,2].>=1) .& (Xk[:,2].<=Mesh.n[2]) .& (Xk[:,3].>=1) .& (Xk[:,3].<=Mesh.n[3]));
	Xk = Xk[II,:];

	ColIdx_k = loc2cs3D(Xk[:,1],Xk[:,2],Xk[:,3],Mesh.n);
	# Rk = sparse(II,ColIdx_k,ones(size(II)),volume,volume);
	# SRk = S*Rk;

	### IF doing SR
	if calcSR == 1
		II .-= 1;
		II = div.(II,n[1]*n[2]*samplingBinning).+1; ## here we "bin"
		globalRowIdx[(offset+1):(offset + length(II))] = II .+ (k-1)*div(n[3],samplingBinning);
	else ### IF doing R
		globalRowIdx[(offset+1):(offset + length(II))] = II .+ (k-1)*volume;
	end
	globalColIdx[(offset+1):(offset + length(II))] = ColIdx_k;
	offset = offset + length(II);
end
# println("Done collecting idxs for matrix");
X = 0;
II = globalRowIdx .> 0;
globalRowIdx = globalRowIdx[II];
globalColIdx = globalColIdx[II];
II = 0;
V = ones(Float32,size(globalColIdx));
V[:] .= prod(Mesh.h);
## We now define (SR) transposed
if calcSR == 1
	# println("Defining the sparse matrix");
	RT = sparse(globalColIdx,globalRowIdx,V,volume,div(num_rotations*n[3],samplingBinning));
	# println("Done defining the sparse matrix");
else
	RT = sparse(globalColIdx,globalRowIdx,V,volume,num_rotations*volume);
end
# println("Done generating matrix");
return RT;
end
