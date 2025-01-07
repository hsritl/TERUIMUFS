clear;
clc;
addpath(genpath(pwd));

dataName='BBCSport_missing20%';
load (dataName)
view_num = length(X);
maxiter = 50;
omega=[1 1 1 1 1 1];
alpha = 10;
beta=1;
tau=100;
pp=1;

% If the dataset is complete, this line can be used to construct unbalanced incomplete dataset
%[X,missing_num,X_missing,zero_indices,one_indices]=construct_UIMdata(Y,20); % Y{view_idx}:n*dv

% The construction of indicator matrices for missing sample
XF=[];
class_num = length(unique(labels));
instance_num=size(labels,2);
for view_idx=1:view_num
    XF=[XF;X{view_idx}];
    W{view_idx} = eye(instance_num);
    W{view_idx}(one_indices{view_idx},:) = [];
end
% UFS stage
tic;
[rank] = TERUIMUFS(X,XF,W,missing_num,class_num,maxiter,alpha,beta,tau,omega,pp);
disp(num2str(toc));
% Clustering stage
XFT=XF';
dim_num = size(XF,1); %The number of total features.
tic;
for t1=1:9
    prop = (t1+1)*0.05; %The proportion of selected features.
    Xsub = XFT(: , rank(1 : floor(prop*dim_num)));
    [res] = litekmeans(Xsub, class_num, 'Replicates',20);
    clear Xsub
    R= EvaluationMetrics(labels', res);
    acc(t1)=R(1);
    nmi(t1)=R(2);
    pu(t1)=R(3);
    fs(t1)=R(4);
    ari(t1)=R(7);
end
disp(num2str(toc));

accx=max(acc);
nmix=max(nmi);
pux=max(pu);
fsx=max(fs);
arix=max(ari);

fprintf('result with the optimal feature selection percentage while missing ratio 0.2: ACC%f, NMI%f, Purity%f, Fscore%f, ARI%f\n', 100*accx, 100*nmix, 100*pux, 100*fsx, 100*arix);
