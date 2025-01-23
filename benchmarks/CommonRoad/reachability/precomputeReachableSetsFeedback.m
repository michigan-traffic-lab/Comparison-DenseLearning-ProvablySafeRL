function precomputeReachableSetsFeedback(name,f,o,param,settings,options)

    % initialize file path
    path = fullfile(fileparts(which(mfilename())),'data',name);
    
    if isdir(path)
        w = warning();
        warning('off');
        rmdir(path,'s');
        warning(w);
    end        
    mkdir(path);

    % loop over all template sets
    for k = 1:length(param.initSet)
        
        % compute feedback matrix
        K = computeFeedbackMatrix(f,param,k);

        % construct nonlinear system object
        [temp,nx] = numberOfInputs(f,3);
        nu = temp(2);
        nw = temp(3);
    
        sys = nonlinearSys([name,'Feedback'], ...
                           @(x,u) [f(x(1:nx),x(2*nx+1:2*nx+nu) + K*(x(1:nx)-x(nx+1:2*nx)), u); ...
                                   f(x(nx+1:2*nx),x(2*nx+1:2*nx+nu),zeros(nw,1)); ...
                                   zeros(nu,1)]);

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
        tmp = polyZonotope(T*R0);
        R0 = cartProd(polyZonotope([tmp.c;tmp.c],[tmp.G;tmp.G],[],tmp.E,tmp.id) + ...
                        cartProd(param.V,zeros(nx,1)),U_); 
        params = [];                
        params.R0 = polyZonotope(R0.c,R0.G,R0.GI, ...
                  [R0.E;zeros(nu_*(settings.N-1),size(R0.E,2))]);

        % reachability parameter
        dt = settings.tFinal/settings.N;
        
        params.U = zonotope(param.W);
        params.tFinal = dt;
        
        % compute reachable sets
        R = [];

        for i = 1:settings.N
            evalc('Rtmp = reach(sys,params,options)');
            R = add(R, Rtmp);
            set = project(Rtmp.timePoint.set{end},1:2*nx);
            params.R0 = cartProd(set,U_);
        end 
        
        % compute set of applied control inputs
        Uint = [];
        
        for i = 1:length(R)
            for j = 1:length(R(i).timeInterval.set)
                set = R(i).timeInterval.set{j};
                Uint = interval(project(set,2*nx+1:2*nx+nu) + ...
                               K*([eye(nx),-eye(nx)]*project(set,1:2*nx)));
            end
        end
        
        disp(['Set ',num2str(k),'   ----------------------------------']);
        Uint

        % compute occupancy sets
        tay = taylorOccupancySet(o,2*nx+nu,dim(param.D));
        ind = nx+1:nx+nu_*settings.N;
        names = param.states(noninvStates);

        cnt_dict = [];
        cnt = 1;
        for i = 1:length(R)
            for j = 1:length(R(i).timeInterval.set)
                cnt_dict(i,j) = cnt;
                cnt = cnt + 1;
            end
        end

        for i = 1:length(R)
            for j = 1:length(R(i).timeInterval.set)
                O = compOccupancySet(R(i).timeInterval.set{j},param.D,tay);   
                exportMotionPrimitive(path,k,cnt_dict(i,j),O,param.initSet{k},param.U, ...
                                      noninvStates,ind,1:i*nu_,names, ...
                                      dt,settings.N);
                %cnt = cnt + 1;
            end
        end
    end
end


% Auxiliary Functions -----------------------------------------------------

function K = computeFeedbackMatrix(f,param,i)
% compute the feedback matrix for the feedback controller

    % linearize system at the center of the intial set
    n = length(param.states);
    m = dim(param.U);
    p = dim(param.W);

    x0 = zeros(n,1);
    x0(setdiff(1:n,param.invStates)) = center(param.initSet{i});
    u0 = center(param.U);
    w0 = center(param.W);
    
    x = sym('x',[n,1]);
    u = sym('u',[m,1]);
    w = sym('w',[p,1]);
    
    Asym = jacobian(f(x,u,w),x);
    Bsym = jacobian(f(x,u,w),u);
    
    A = eval(subs(Asym,[x;u;w],[x0;u0;w0]));
    B = eval(subs(Bsym,[x;u;w],[x0;u0;w0]));
    
    % compute feedback matrix using LQR approach
    Q = eye(n);
    R = eye(m);
    
    K = -lqr(A,B,Q,R);
end

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
    c = O.c; G = O.G; E = O.E; Grest = O.GI;
    
    if j == 80
        rest = interval(zonotope(0*c,Grest));
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
           E = [E; set.E(tmp(1),:)];
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
           E = [E; set.E(tmp(1),:)];
           index = [index, tmp];
       else
           E = [E; zeros(1, size(E,2))]; 
       end
    end
    
    E = [E; zeros(length(ind2)-length(ind3),size(E,2))];
    
    tmp = setdiff(1:size(set.E,1),index);
    E = [E; set.E(tmp,:)];
    
    % remove all dependent factors that do not belong to non-invariant
    % states or inputs
    tmp = length(ind1) + length(ind2);
    ind = find(sum(E(tmp+1:end,:),1) > 0);
    ind_ = setdiff(1:size(E,2),ind);
    
    pZ = polyZonotope(set.c,set.G(:,ind),set.GI,E(:,ind));
    Z = zonotope(pZ);
    Z = reduce(Z,'girard',20);

    set = polyZonotope(center(Z),set.G(:,ind_),generators(Z),E(1:tmp,ind_));
end