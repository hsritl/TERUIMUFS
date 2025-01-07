function [ranking] = TERUIMUFS(X,XF,W,missing_num,class_num,maxiter,alpha,beta,tau,omega,pp)
view_num=size(X,2);
instance_num=size(X{1},2);
for view_idx=1:view_num
    dim_num(view_idx)=size(X{view_idx},1);
end
gamma = 10000;
sX = [instance_num, instance_num, view_num];
mu1 = 0.1;
mu2 = 0.1;
mu3 = 0.1;
mu4 = 0.1;
mu_max = 1e10;
eta = 2;
epson = 1e-7;
converge_J1=[];
converge_J2=[];
converge_J3=[];
converge_J4=[];

% Initialization of lambda
lambda = 1 / view_num * ones(view_num,1);
% Initialize similarity matrix A
A = cell(1,view_num);
for i = 1:view_num
    sigma = optSigma(X{i}')^2; % Median of Euclidean distance between all two samples
    A{i} = constructW(X{i}', struct('k',5, 'WeightMode', 'HeatKernel', 't', sigma));
    A{i} = view_num * A{i} ./ repmat( sum(A{i},2 ) , 1 , size(A{i},1)); %Row normalization and multiply by V
end
% Initialize AC, L
AC = zeros(size(A{1}));
for i = 1:view_num
    AC = AC + lambda(i) * A{i};
end
SS = (AC + AC') / 2;
SSS = diag(sum(SS));
L = SSS - SS;
% Initialize cluster indicator matrix H
options.method = 'k_means';
H = init_H(XF',class_num,options);
% Initialize others
C = cell(1,view_num);
B = cell(1,view_num);
D = cell(1,view_num);
E = cell(1,view_num);
XN = cell(1,view_num);
J1 = cell(1,view_num);
J2 = cell(1,view_num);
J3 = cell(1,view_num);
J4 = cell(1,view_num);
B1 = cell(1,view_num);
B2 = cell(1,view_num);
Z = cell(1,view_num);
Y = cell(1,view_num);
Ktemp=cell(1,view_num);
for i=1:view_num
    C{1,i} = eye(dim_num(i),class_num);
    B{1,i} = C{1,i}' * X{1,i} * H;
    D{1,i} = eye(dim_num(i));
    E{i}=zeros(dim_num(i),missing_num(i));
    XN{i}=X{i};
    J1{i} = zeros(dim_num(i),instance_num);
    J2{i} = zeros(instance_num,instance_num);
    J3{i} = zeros(instance_num,instance_num);
    J4{i} = zeros(dim_num(i),instance_num);
    B1{i}=zeros(dim_num(i),instance_num);
    B2{i}=zeros(instance_num,instance_num);
    Z{i}=A{i};
    Y{i}=A{i};
end

for iter = 1:maxiter
    H1H=[];
    H2H=[];
    % Update lambda
    lambda = solve_delta(AC,A);
    % Update C, D
    for i = 1:view_num
        temp_W = (XN{1,i} * XN{1,i}') + tau * (X{1,i} * L * X{1,i}') + alpha * D{1,i};
        C{1,i} = temp_W \ XN{1,i} * H * B{1,i}';
        tempD = 0.5*pp * (sqrt(sum(C{1,i}.^2,2) + eps)).^(pp-2);
        D{1,i} = diag(tempD);
        F8{1,i}=C{1,i}'*X{1,i};
    end
    % Update B
    for i = 1:view_num
        SVD = C{1,i}'*XN{1,i}*H; %This corresponds to the transpose in the paper!
        [V_B,~,U_B] = svd(SVD,'econ');
        B{1,i} = V_B * U_B';
    end
    % Update Z
    HN = max(H,0);
    % Update H
    SVD = zeros(size(HN));
    for i = 1:view_num
        SVD = SVD + XN{1,i}' * C{1,i} * B{1,i} ;
    end
    SVD = SVD + gamma * HN; %This corresponds to the transpose in the paper!
    [V_H,~,U_H] = svd(SVD,'econ');
    H = V_H * U_H';
    % Update S
    AC = Update_S(A,F8,view_num,1,tau,lambda);
    SS = (AC + AC') / 2;
    SSS = diag(sum(SS));
    L = SSS - SS;
    % Update E
    for i = 1:view_num
        F1=X{i}-XN{i}+J4{i}/mu4;
        F2=W{i}-W{i}*Z{i};
        F3=X{i}-X{i}*Z{i}-B1{i}+J1{i}/mu1;
        E_l=-mu4*F1*W{i}'-mu1*(F3*F2');
        E_r=2*beta*ones(missing_num(i),1)*ones(1,missing_num(i))+(-2*beta+mu4)*eye(missing_num(i),missing_num(i))+mu1*(F2*F2');
        E{i}=E_l/E_r;
        Ktemp{i}=X{i}+E{i}*W{i};
    end
    % Update XN
    for i = 1:view_num
        F4=B{i}*H';
        F5=X{i}+E{i}*W{i}+J4{i}/mu4;
        XN_l=2*C{i}*C{i}'+mu4*eye(dim_num(i),dim_num(i));
        XN_r=2*C{i}*F4+mu4*F5;
        XN{i}=XN_l\XN_r;
    end
    % Update B1,B2
    for view_idx=1:view_num
        H1{view_idx}=-Ktemp{view_idx}*Z{view_idx}+Ktemp{view_idx}+J1{view_idx}/mu1;
        H1H=[H1H;H1{view_idx}];
    end
    B1B = solve_l1l2(H1H,1/mu1);
    for view_idx=1:view_num
        if view_idx==1
            B1{view_idx}=B1B(1:dim_num(view_idx),:);
            B1_res=B1B(dim_num(view_idx)+1:size(B1B,1),:);
        else
            B1{view_idx}=B1_res(1:dim_num(view_idx),:);
            B1_res=B1_res(dim_num(view_idx)+1:1:size(B1_res,1),:);
        end
    end
    for view_idx=1:view_num
        H2{view_idx}=Z{view_idx}-A{view_idx}+J2{view_idx}/mu2;
        H2H=[H2H;H2{view_idx}];
    end
    B2B = solve_l1l2(H2H,1/mu2);
    for view_idx=1:view_num
        if view_idx==1
            B2{view_idx}=B2B(1:instance_num,:);
            B2_res=B2B(instance_num+1:size(B2B,1),:);
        else
            B2{view_idx}=B2_res(1:instance_num,:);
            B2_res=B2_res(instance_num+1:1:size(B2_res,1),:);
        end
    end
    % Update A
    for i = 1:view_num
        F6=Z{i}-B2{i}+J2{i}/mu2;
        F7=-Y{i}+J3{i}/mu3;
        A_temp=2*alpha*AC+mu2*F6-mu3*F7;
        A{i}=A_temp/(2*alpha*alpha+mu2+mu3);
    end
     % Update Z^v
    for view_idx=1:view_num
        F9=X{view_idx}+E{view_idx}*W{view_idx}-B1{view_idx}+J1{view_idx}/mu1;
        F10=-A{view_idx}-B2{view_idx}+J2{view_idx}/mu2;
        Z_left=mu1*Ktemp{view_idx}'*Ktemp{view_idx}+mu2*eye(instance_num,instance_num);
        Z_right=mu1*Ktemp{view_idx}'*F9-mu2*F10;
        Z{view_idx}=Z_left\Z_right;
    end
     % Update Y^v
    A_tensor = cat(3, A{:,:});
    J3_tensor = cat(3, J3{:,:});
    aT = A_tensor(:);
    j3T = J3_tensor(:);
    [yT, ~] = wshrinkObj(aT+1/mu3*j3T,1/mu3,sX,0,3,omega);
    Y_tensor = reshape(yT, sX);
    for view_idx=1:view_num
        Y{view_idx} = Y_tensor(:,:,view_idx);
    end
    % Update J1234
    for view_idx=1:view_num
        J1{view_idx} = J1{view_idx}+mu1*(Ktemp{view_idx}-Ktemp{view_idx}*Z{view_idx}-B1{view_idx});
        J2{view_idx} = J2{view_idx}+mu2*(Z{view_idx}-A{view_idx}-B2{view_idx});
        J3{view_idx} = J3{view_idx}+mu3*(A{view_idx}-Y{view_idx});
        J4{view_idx} = J4{view_idx}+mu4*(Ktemp{view_idx}-XN{view_idx});
    end
    % Update mu
    mu1  = min(mu_max,mu1*eta);
    mu2  = min(mu_max,mu2*eta);
    mu3  = min(mu_max,mu3*eta);
    mu4  = min(mu_max,mu4*eta);
end

%Calculate feature ranking
WW = [];
for i = 1:view_num
    WW = [WW;C{1,i}];
end
[~,ranking] = sort(sum(WW.*WW,2),'descend');
ttt=sum(WW.*WW,2);
ranking=ranking';
end

