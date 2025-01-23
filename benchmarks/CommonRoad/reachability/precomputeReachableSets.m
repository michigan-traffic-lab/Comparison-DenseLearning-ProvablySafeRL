function precomputeReachableSets(name,f,o,param,settings,options)

    % initialize file path
    path = fullfile(fileparts(which(mfilename())),'data',name);
    
    if isdir(path)
        w = warning();
        warning('off');
        rmdir(path,'s');
        warning(w);
    end        
    mkdir(path);

    % construct nonlinear system object
    [temp,nx] = numberOfInputs(f,3);
    nu = temp(2);
    
    sys = nonlinearSys(name,@(x,u) [f(x,x(nx+1:nx+nu),u);zeros(nu,1)]);

    % loop over all template sets
    parfor k = 1:length(param.initSet)

        disp(['Set ',num2str(k),'   ----------------------------------']);
        
        % construct initial set
        noninvStates = setdiff(1:nx,param.invStates);
        T = zeros(nx);
        R0 = cartProd(zeros(length(param.invStates),1),param.initSet{k});
        
        for i = 1:length(param.invStates)
            T(param.invStates(i),i) = 1;
        end
        
        for i = 1:length(noninvStates)
            T(noninvStates(i),length(param.invStates)+i) = 1; 
        end
        
        U_ = polyZonotope(param.U);
        nu_ = length(U_.id);
        R0 = cartProd(polyZonotope(T*R0)+param.V,U_); 
        params = [];
        params.R0 = polyZonotope(R0.c,R0.G,R0.Grest, ...
                  [R0.expMat;zeros(nu_*(settings.N-1),size(R0.expMat,2))]);

        % reachability parameter
        dt = settings.tFinal/settings.N;
        
        params.U = zonotope(param.W);
        params.tFinal = dt;
        
        % compute reachable sets
        R = [];

        for i = 1:settings.N
            Rtmp = reach(sys,params,options);
            R = add(R, Rtmp);
            set = project(Rtmp.timePoint.set{end},1:nx);
            params.R0 = cartProd(set,U_);
        end 

        % compute occupancy sets
        tay = taylorOccupancySet(o,nx+nu,dim(param.D));
        cnt = 1;
        ind = nx+1:nx+nu_*settings.N;
        names = param.states(noninvStates);

        for i = 1:length(R)
            for j = 1:length(R(i).timeInterval.set)
                O = compOccupancySet(R(i).timeInterval.set{j},param.D,tay);   
                exportMotionPrimitive(path,k,cnt,O,param.initSet{k},param.U, ...
                                      noninvStates,ind,1:i*nu_,names, ...
                                      dt,settings.N);
                cnt = cnt + 1;
            end
        end
    end
end


% Auxiliary Functions -----------------------------------------------------

function O = compOccupancySet(R,D,tay)
% compute the occupancy set by adding the car dimensions to the reachable
% set    

    % evaluate derivatives at linearization point
    X = cartProd(R,polyZonotope(D));
    p = center(X);

    [f,A,Q,T] = evalDerivatives(X,p,tay);

    % compute Largrange remainder
    rem = lagrangeRemainder(X,p,T);

    % compute over-approximating polynmoial zonotope
    O = f + exactPlus(A * (X + (-p)), 0.5*quadMap((X + (-p)),Q)) + rem;
end

function tay = taylorOccupancySet(o,nx,nd)
% compute the symbolic derivatives of the Taylor expansion of the nonlinear
% function used to compute the occupancy set

    % function handle for the nonlinear function    
    x = sym('x',[nx,1]);
    d = sym('d',[nd,1]);
    z = [x; d];
    
    f = o(x,d);
    fun =  matlabFunction(f,'Vars',{z});
    
    % first-order derivative
    A = jacobian(f,z);
    Afun =  matlabFunction(A,'Vars',{z});
    
    % second order derivative
    Qfun = cell(length(f),1);
    for i = 1:length(f)
       temp = hessian(f(i),z); 
       Qfun{i} =  matlabFunction(temp,'Vars',{z});
    end
    
    % Lagrange remainder
    Tfun = cell(size(A));
    for i = 1:size(A,1)
        for j = 1:size(A,2)
            temp = hessian(A(i,j),z);
            if any(any(temp ~= 0))
                Tfun{i,j} = matlabFunction(temp,'Vars',{z});
            end
        end
    end
    
    % store function handles
    tay.fun = fun; tay.Afun = Afun; tay.Qfun = Qfun; tay.Tfun = Tfun;
end

function [f,A,Q,T] = evalDerivatives(X,p,tay)
% evaluate the derivatives at the linearization point

    % interval enclosure of the set
    int = interval(X);
    
    f = tay.fun(p);
    A = tay.Afun(p);
    
    Q = cell(length(f),1);
    for i = 1:length(f)
       funHan = tay.Qfun{i};
       Q{i} = funHan(p);
    end
    
    T = cell(size(A));
    for i = 1:size(A,1)
        for j = 1:size(A,2)
            if ~isempty(tay.Tfun{i,j})
                funHan = tay.Tfun{i,j};
                T{i,j} = funHan(int);
            end
        end
    end
end

function rem = lagrangeRemainder(X,p,T)
% comptute the Lagrange remainder of the Taylor series

    % interval enclousre of the shifted initial set
    int = interval(X) - p;

    % Lagrange remainder term
    rem = interval(zeros(size(T,1),1));
    
    for i = 1:size(T,1)
        for j = 1:size(T,2)
            if ~isempty(T{i,j})
                rem(i) = rem(i) + int(j) * transpose(int) * T{i,j} * int;
            end
        end
    end
    
    % convert to zonotope
    rem = zonotope(1/6*rem);
end

function exportMotionPrimitive(path,i,j,O,R0,U,ind1,ind2,ind3,names,dt,N)
% export the precomputed motion primitives to files    

    % upper and lower bounds of the initial set
    l = infimum(R0);
    u = supremum(R0);
    
    % center and generator matrix of the set of input commands
    U = zonotope(U);
    c_u = center(U);
    G_u = generators(U);
    
    % vertices of the polygon enclosure of the occupancy set
    pgon = polygon(O);
    %pgon = simplify(pgon,0.01);
    V = pgon.set.Vertices';
    
    % properties of the polynomial zonotope
    O = getFactors(O,ind1,ind2,ind3);
    c = O.c; G = O.G; E = O.expMat; Grest = O.Grest;
    
    if j == 10
        rest = interval(zonotope(0*c,Grest))
    end

    % export data
    filename = fullfile(path,['reach_',num2str(i),'_',num2str(j)]);
    save(filename,'c','G','E','Grest','V','l','u','c_u','G_u', ...
                                                        'names','dt','N');
end

function set = getFactors(set,ind1,ind2,ind3)
% bring the polynomial zonotope to the correct order based on the factor ID

    % fill up non-existing exponents with zero for non-invariant states
    E = []; index = [];

    for i = 1:length(ind1)
       tmp = find(set.id == ind1(i));
       if ~isempty(tmp)
           E = [E; set.expMat(tmp(1),:)];
           index = [index, tmp];
       else
           E = [E; zeros(1, size(E,2))]; 
       end
    end

    % fill up non-existing exponents with zero for states corr. to inputs
    ind2_ = ind2(ind3);
    
    for i = 1:length(ind2_)
       tmp = find(set.id == ind2_(i));
       if ~isempty(tmp)
           E = [E; set.expMat(tmp(1),:)];
           index = [index, tmp];
       else
           E = [E; zeros(1, size(E,2))]; 
       end
    end
    
    E = [E; zeros(length(ind2)-length(ind3),size(E,2))];
    
    tmp = setdiff(1:size(set.expMat,1),index);
    E = [E; set.expMat(tmp,:)];
    
    % remove all dependent factors that do not belong to non-invariant
    % states or inputs
    tmp = length(ind1) + length(ind2);
    ind = find(sum(E(tmp+1:end,:),1) > 0);
    ind_ = setdiff(1:size(E,2),ind);
    
    pZ = polyZonotope(set.c,set.G(:,ind),set.Grest,E(:,ind));
    Z = zonotope(pZ);
    Z = reduce(Z,'girard',20);

    set = polyZonotope(center(Z),set.G(:,ind_),generators(Z),E(1:tmp,ind_));
end