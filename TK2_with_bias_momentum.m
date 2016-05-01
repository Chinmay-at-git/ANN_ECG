% ANN Takehome 2
%clear all;
%clear all;
close all;
N=0; % Epoch Iteration Counter
Nmax= 15000;
egraph=zeros(Nmax+2,1);
TRAIN(:,28)=1;  %Bias adding
SIZE_TRAIN = size(TRAIN,1);
LRATE = 0.0006/SIZE_TRAIN;
W_1= -0.18 + 0.184*2*rand(28,55);
W_2= -0.09 + 0.09*2*rand(55,1);% To Avoid Saturations! 
%W_1= rand(27,55);
%W_2=rand(55,1);
initW_1=W_1;
initW_2=W_2;
W_1=ROMW_1;
W_2=ROMW_2;
OP1= zeros(28,1);
OP2= zeros(55,1);
OP3= 0;
delta1 = zeros(28,55);
delta1_old=delta1;
alpha=0.52;
delta_p1 = zeros(28,55);
delta2 = zeros(55,1);
delta2_old=delta2;
delta_pop_old=zeros(SIZE_TRAIN,1);
delta_p2= zeros(55,1);
%AEP = importdata('S:\Spring2016\ANN\takehome2\AEP.txt');
%NOAEP = importdata('S:\Spring2016\ANN\takehome2\NOAEP.txt');
%NTRAIN = NOAEP; % To be changed!!

%TRAIN=NTRAIN;
EPs= zeros(SIZE_TRAIN,1);
Es= zeros(Nmax,1);
OPs=zeros(SIZE_TRAIN,1);
deltas_pop=zeros(SIZE_TRAIN,1);
deltas_1=zeros(SIZE_TRAIN,1);

%TRAIN_T=     ones(SIZE_TRAIN,1);
%TRAIN_T =    0.9 * TRAIN_T; % To bechanged and be part of Dataset partitioning
while true
    if N > Nmax
        break;
    end
    N= N + 1;
    
    OP2s=tanh(TRAIN*W_1);
    


    OP3s=tanh(OP2s*W_2);
    EPs = 0.5* (TRAIN_T - OP3s).*(TRAIN_T - OP3s);
    delta_pop = (TRAIN_T - OP3s) .* (ones(SIZE_TRAIN,1)-OP3s.*OP3s);
    delta2 = delta2 + LRATE * ( OP2s'*delta_pop) +alpha*delta2_old;
    delta_p2 = (ones(SIZE_TRAIN,55)-OP2s .*OP2s) .*(delta_pop*(W_2'));
    delta1 = delta1 + LRATE * (TRAIN' * delta_p2) + alpha*delta1_old;
    Es(N)= sum(EPs);
    delta1_old=delta1;
    delta2_old=delta2;
    % Time to correct weights
    W_2 = W_2 + delta2 ;
    delta2 = delta2*0;
    W_1 = W_1 + delta1;
    delta1 = delta1 * 0;
        
zer=0;
tp=0;
tn=0;

fp=0;
fn=0;
graph=zeros(SIZE_TRAIN,1);

%for i=1:SIZE_TRAIN
%    if OP3s(i)== 0 
%        zer = zer +1 ;
%        graph(i)=0;
%        continue;
%    end
%    if OP3s(i)>0 && TRAIN_T(i)>0
%        tp = tp + 1;
%        graph(i) = 1;
%        continue;
%    end
%    if OP3s(i)<0 && TRAIN_T(i)<0
%        tn = tn + 1;
%        graph(i) = -1;
%    else
%        fp = fp + 1;
%        graph(i) = 0.5;
%    end
%end
egraph(N)=tp+tn;
    %Continue to next epoch
end
plot(1:Nmax+1,Es);
%figure, plot(1:SIZE_TRAIN,EPs);


%% Test Data Checks:
test_tp=0; %class 1
test_tn=0; %class 2
test_zer=0;
fp=0;
fn=0;

TEST=[AEP_TEST(:,1:27);NOAEP_TEST(:,1:27)];
TEST(:,28)=1;
SIZE_TEST= size(TEST,1);
TEST_T=[ones(size(AEP_TEST,1),1) ; -1*ones(size(NOAEP_TEST,1),1) ];
OP2s=tanh(TEST*W_1);
OP3s=tanh(OP2s*W_2);
for i=1:SIZE_TEST
   if TEST_T(i)>0 && OP3s(i)>0
        test_tp=test_tp+1;
   end
   if TEST_T(i)<0 && OP3s(i) <0
        test_tn=test_tn+1;
   end
   if OP3s(i) == 0
        test_zer=test_zer+1;
   end
   if TEST_T(i)<0 && OP3s(i)>0
	fp=fp+1;
	end
 if TEST_T(i)>0 && OP3s(i)<0
	fn=fn+1;
end

	        
end
test_zer
sensitivity = test_tp/(test_tp+fn)
specificity  = test_tn / (test_tn+fp)
ETSS=0.5*(TEST_T-OP3s)'*(TEST_T-OP3s)
%for P=1:SIZE_TRAIN
%    for i=1:55
%        net=0;
%        for j=1:27
%           net =  net + TRAIN(P,j)*W_1(j,i); 
%        end
%        %net = net/27; % Makes Weights simple
%        
%        OP2(i) =  tanh(net); %Activation Function
%       % if OP2(i)==1 || OP2(i)==-1
%       % OP2(i) = OP2(i) - 0.01*OP2(i);
%       % end
%    end
%
%    net = 0;
%    for i=1:55
%        net = net + OP2(i)*W_2(i,1);
%    end
%    %net = net / 55; % Makes Weights simple
%    net;
%    OP3 = tanh(net);
%    OPs(P)=OP3;
%    
