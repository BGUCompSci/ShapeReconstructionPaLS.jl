using VolReconstruction.Utils
using VolReconstruction.DipTransform
ENV["MPLBACKEND"] = "Qt4Agg"

plotting = true;
if plotting
	using jInvVis
	using PyPlot
end


println("Rotation operator transpose test");
n = [65,65,65];
u = getDiamondModel(n);
u = convert(Array{Float64}, u)

if plotting
	figure()
	plotModel(u)
end

theta_phi = deg2rad.([37.0;24.0]);
display(theta_phi)
b = [1.0;2.0;3.0];
# b[:] = 0.0;


# RT = generateSamplingMatrix(u,theta_phi,1,1)
# v = RT'*u[:]
# println("The norm is:")
# println(norm(RT*v - u[:]))

v,xt,xtt = rotateAndMove3D(u,theta_phi,b,false);

if plotting
	figure()
	plotModel(v)
end
#uu = rotateAndMove3DTranspose(v,theta_phi,b);
uu,xt,xtt = rotateAndMove3D(v,theta_phi,b,true);
if plotting
	figure()
	plotModel(uu)
end

println(vecdot(v,v))
println(vecdot(uu,u));

# if abs(vecdot(v,v) - vecdot(uu,u)) > 1e-5
# 	error("Bug");
# end
print(norm(uu[:]-u[:]))
if norm(uu[:]-u[:]) > 1e-5
	error("Bug");
end
