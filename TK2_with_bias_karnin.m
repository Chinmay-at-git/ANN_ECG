% ANN Takehome 2
%clear all;
%clear all;
close all;
N=0; % Epoch Iteration Counter
Nmax= 10000;
egraph=zeros(Nmax+2,1);
TRAIN(:,28)=1;  %Bias adding
SIZE_TRAIN = size(TRAIN,1);
LRATE = 0.00005/SIZE_TRAIN;
W_1= -0.18 + 0.184*2*rand(28,55);
W_2= -0.09 + 0.09*2*rand(55,1);% To Avoid Saturations! 
%W_1= rand(27,55);
%W_2=rand(55,1);
%initW_1=W_1;
%initW_2=W_2;
W_1=ROMW_1;
W_2=ROMW_2;
OP1= zeros(28,1);
OP2= zeros(55,1);
OP3= 0;
delta1 = zeros(28,55);
delta_p1 = zeros(28,55);
delta2 = zeros(55,1);
delta_p2= zeros(55,1);
%AEP = importdata('S:\Spring2016\ANN\takehome2\AEP.txt');
%NOAEP = importdata('S:\Spring2016\ANN\takehome2\NOAEP.txt');
%NTRAIN = NOAEP; % To be changed!!

%TRAIN=NTRAIN;
EPs= zeros(SIZE_TRAIN,1);
Es= zeros(Nmax,1);
OPs=zeros(SIZE_TRAIN,1);

S= zeros(28,55);
W_init=W_1;
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
    delta2 = delta2 + LRATE * ( OP2s'*delta_pop);
    delta_p2 = (ones(SIZE_TRAIN,55)-OP2s .*OP2s) .*(delta_pop*(W_2'));
    delta1 = delta1 + LRATE * (TRAIN' * delta_p2);
    Es(N)= sum(EPs);
    S= S + delta1/(LRATE*SIZE_TRAIN);
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
%% Prune the Network : Karnin
THR=LRATE; %Not using THR
S_old=S;
S= S .* (W_1./(W_1-W_init));
%imagesc(S);
pruned=0;
reshapeS=reshape(abs(S),[28*55 1]);
[sortedS indS]=sort(reshapeS);
toprune=zeros(28*55/20,1);
for i=1:28*55/20
    toprune(i,1)=reshapeS(indS(i));
    reshapeS(indS(i))=0;
end
prunedl=zeros(28*55,1);
S=reshape(reshapeS,[28 55]);
for i=1:28
    for j=1:55
        if(S(i,j)== 0)
            prunedl(pruned+1)=W_1(i,j);
            W_1(i,j)=0;
            
            pruned=pruned+1;
        end
    end
end

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