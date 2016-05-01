% 
clear all;
close all;
AEP = importdata('AEP.txt');
NOAEP = importdata('NOAEP.txt');
AEP(:,28)=0.9*ones(size(AEP,1),1);
NOAEP(:,28)=-0.9*ones(size(NOAEP,1),1);
AEP_rand= randperm(size(AEP,1));
AEP_TRAIN = AEP;%AEP(AEP_rand(1:67),:);
AEP_TEST=AEP(AEP_rand(61:83),:);
AEP_SO(:,28)=0.9;
NOAEP_rand = randperm(size(NOAEP,1));
NOAEP_TRAIN = NOAEP(NOAEP_rand(1:268),:);
NOAEP_TEST=NOAEP(NOAEP_rand(1001:1500),:);


TRAIN = [NOAEP_TRAIN(1:40,:);AEP_TRAIN(1:40,:)];
%TRAIN_rand = randperm(size(TRAIN,1)/2);
NTRAIN=TRAIN;
j=1;
for i=1:size(TRAIN,1)/2
    NTRAIN(j,:)=TRAIN(i,:);
    NTRAIN(j+1,:)=TRAIN(i+size(TRAIN,1)/2,:);
    j=j+2;
end
TRAIN=NTRAIN;
%TRAIN = TRAIN(TRAIN_rand(1:size(TRAIN,1)),:);
TRAIN_T = TRAIN(:,28);
TRAIN= TRAIN(:,1:27);
NTRAIN=TRAIN;
for i=1:27
    TRAIN(:,i)= -1 + 2* ((NTRAIN(:,i)-min(NTRAIN(:,i)))/(max(NTRAIN(:,i))-min(NTRAIN(:,i))));
end
%TRAIN(:,28)=1;
SIZE_TRAIN=size(TRAIN,1);