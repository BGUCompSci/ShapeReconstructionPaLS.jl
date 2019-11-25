
function loc2cs3D(loc1::Union{Int64,Array{Int64}},loc2::Union{Int64,Array{Int64}},loc3::Union{Int64,Array{Int64}},n::Array{Int64,1})
@inbounds cs = loc1 .+ (loc2.-1)*n[1] .+ (loc3.-1)*n[1]*n[2];
return cs;
end



function generateProjectionOperator(param::SfSParam)
Mesh = param.ObjMesh;
CamLoc = param.CamLoc;
XScreenLoc = param.XScreenLoc;
ScreenMesh = param.ScreenMeshX2X3;
theta_phi = param.theta_phi_rad;

X = getCellCenteredGrid(Mesh);
mid_domain = [(Mesh.domain[1]+Mesh.domain[2])/2,(Mesh.domain[3]+Mesh.domain[4])/2,(Mesh.domain[5]+Mesh.domain[6])/2];
x_mesh = (Mesh.domain[1] + 0.5*Mesh.h[1]):Mesh.h[1]:(Mesh.domain[2] - 0.5*Mesh.h[1]);

X[:,1] .= X[:,1] .- mid_domain[1];
X[:,2] .= X[:,2] .- mid_domain[2];
X[:,3] .= X[:,3] .- mid_domain[3];


X_screen = getCellCenteredGrid(ScreenMesh);
midScreen = [(ScreenMesh.domain[1]+ScreenMesh.domain[2])/2,(ScreenMesh.domain[3]+ScreenMesh.domain[4])/2];
X_screen[:,1] .-= midScreen[1];
X_screen[:,2] .-= midScreen[2];

half_screen_domain_length = [(ScreenMesh.domain[2]-ScreenMesh.domain[1])/2,(ScreenMesh.domain[4]-ScreenMesh.domain[3])/2];

offset = 0;
n =  Mesh.n;
invh = 1.0 ./Mesh.h;

volume = prod(Mesh.n);
n_tup = tuple(Mesh.n...);
n_screen_tup = tuple(ScreenMesh.n...);
n_screen = ScreenMesh.n;
numelScreen = prod(ScreenMesh.n);

I = ones(Int32,(n[1]*numelScreen));
J = ones(Int32,(n[1]*numelScreen));
V = zeros(Float32,(n[1]*numelScreen));
offset = 0;
g_idx = 1;
for k=1:numelScreen
	## definition of the line between screen point and camera point
	X_screen_k = X_screen[k,:];
	## projecting point Z to the screen:
	# ProjectionLoc(t) = CamLoc + t*(X_screen - CamLoc)
	# We need all the mesh points which are on this line;
	# For a given x location in the mesh, we search for the YZ locations on the line
	
	# For the first coordinate: x_mesh_jj = CamLoc[1] + t*(X_screen - CamLoc[1]);
	for ii = 1:n[1]
		x_mesh_ii = x_mesh[ii];
		t = (x_mesh_ii - CamLoc[1])/(XScreenLoc - CamLoc[1]);
		y_mesh = CamLoc[2] + t*(X_screen[k,1] - CamLoc[2]);
		z_mesh = CamLoc[3] + t*(X_screen[k,2] - CamLoc[3]);
		y_mesh += mid_domain[2];
		z_mesh += mid_domain[3];
		jj = round(Int64,y_mesh*invh[2]+0.5);
		kk = round(Int64,z_mesh*invh[3]+0.5);
		if jj <= n[2] && jj > 0 && kk <= n[3] && kk > 0
			J[g_idx] = loc2cs3D(ii,jj,kk,Mesh.n);
			I[g_idx] = k;
			V[g_idx] = 1.0;
			g_idx += 1;
		end
	end
end
RT = sparse(J,I,V,volume,numelScreen);

return RT;
end



# function generateProjectionOperatorWithRotation(param::SfSParam)
# Mesh = param.ObjMesh;
# CamLoc = param.CamLoc;
# XScreenLoc = param.XScreenLoc;
# ScreenMesh = param.ScreenMeshX2X3;
# theta_phi = param.theta_phi_rad;

# X = convert(Array{Float32,2},getCellCenteredGrid(Mesh));
# mid_domain = [(Mesh.domain[1]+Mesh.domain[2])/2,(Mesh.domain[3]+Mesh.domain[4])/2,(Mesh.domain[5]+Mesh.domain[6])/2];
# X[:,1] = X[:,1] - mid_domain[1];
# X[:,2] = X[:,2] - mid_domain[2];
# X[:,3] = X[:,3] - mid_domain[3];

# midScreen = [(ScreenMesh.domain[1]+ScreenMesh.domain[2])/2,(ScreenMesh.domain[3]+ScreenMesh.domain[4])/2];
# half_screen_domain_length = [(ScreenMesh.domain[2]-ScreenMesh.domain[1])/2,(ScreenMesh.domain[4]-ScreenMesh.domain[3])/2];


# num_rotations = size(theta_phi,1);
# offset = 0;
# n =  Mesh.n;
# volume = prod(Mesh.n);

# n_screen_tup = tuple(ScreenMesh.n...);

# numelScreen = prod(ScreenMesh.n);


# globalRowIdx = zeros(Int32,(volume*num_rotations+5));
# globalColIdx = zeros(Int32,(volume*num_rotations+5));

# calcSR = 1;

# R = 0;
# offset = 0;
# for k=1:num_rotations
	
	# ## Rotating the original locations X.
	# Rk = convert(Array{Float32,2},getRotate3D(deg2rad.(theta_phi[k,1]),deg2rad.(theta_phi[k,2])));
	# Xk = X*Rk';
	# Zk = Xk; Xk = 0;
	
	# ## projecting point Z to the screen:
	# # ProjectionLoc(t) = CamLoc + t*(X - CamLoc)
	# for k=1:3  ## Zk = Xk - CamLoc
		# Zk[:,k] = Zk[:,k] - CamLoc[k];
	# end;
	# # ProjectionLoc(t) at screelLock will have first coordinate equals XScreenLoc
	# tk = (XScreenLoc - CamLoc[1])./Zk[:,1];
	# Sk = zeros(eltype(Zk),size(Zk,1),2);
	# Sk[:,1] = CamLoc[2] + tk.*Zk[:,2];
	# Sk[:,2] = CamLoc[3] + tk.*Zk[:,3];
	
	# Sk[:,1] = (Sk[:,1] + half_screen_domain_length[1])*(1 ./ScreenMesh.h[1]) + 0.5;
	# Sk[:,2] = (Sk[:,2] + half_screen_domain_length[2])*(1 ./ScreenMesh.h[2]) + 0.5;
	# Sk = round.(Int32,Sk);
	# II = find((Sk[:,1].>=1) .& (Sk[:,1].<=ScreenMesh.n[1]) .& (Sk[:,2].>=1) .& (Sk[:,2].<=ScreenMesh.n[2]) );
	# Sk = Sk[II,:];
	# RowIdx_k = Base.sub2ind(n_screen_tup,Sk[:,1],Sk[:,2]);
	# globalColIdx[(offset+1):(offset + length(II))] = II;
	# globalRowIdx[(offset+1):(offset + length(II))] = RowIdx_k;
	# offset = offset + length(II);
# end
# X = 0;
# II = globalRowIdx .> 0;
# globalRowIdx = globalRowIdx[II];
# globalColIdx = globalColIdx[II];
# II = 0;
# V = ones(Float32,size(globalColIdx)); 
# RT = sparse(globalColIdx,globalRowIdx,V,volume,num_rotations*numelScreen);
# return RT;
# end

