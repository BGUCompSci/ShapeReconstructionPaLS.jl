using VolReconstruction.VisualHull;



n = 20;
m = 10;
u = rand(20);
At = sprand(m,n,4.0/m)';
At.nzval[:] = 1.0;

d0,Jt = softMaxProjWithSensMat(At,u);

du0 = 0.01*randn(size(u));

println(norm(maximum((At')*spdiagm(u),2) - d0))

for ii = 1:8
	du = (0.5^ii)*du0;
	ut = u + du;
	dt, = softMaxProjWithSensMat(At,ut);
	println("norm(dt-d0): ",norm(dt-d0),", norm(dt - d0 - J0*dm): ",norm(dt - d0 - Jt'*du));
end

