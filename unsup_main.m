clear all; clc; format compact; close all; format short eng

%%
load deepsig_2016_even_noamsb.mat

% train
n = size(XTrain, 4);

numToRemove = round(n * 0.1724);
indicesToRemove = randperm(n, numToRemove);
% Remove from XTrain
XTrain(:, :, :, indicesToRemove) = [];

% Remove from YTrain
YTrain(indicesToRemove) = [];

% Remove from STrain
STrain(indicesToRemove) = [];

% val
n = size(XValidation, 4);

numToRemove = round(n * 0.1724);
indicesToRemove = randperm(n, numToRemove);
% Remove from XTrain
XValidation(:, :, :, indicesToRemove) = [];

% Remove from YTrain
YValidation(indicesToRemove) = [];

% Remove from STrain
SValidation(indicesToRemove) = [];

%% prototype learning start

% 15% of deepsig 2016 dataset
% XTrain is dataframes, YTrain is modulation labels
% XValidation and YValidation are test set
% STrain and SValidation are SNR labels
load deepsig_2016_even_noamsb.mat

global xproto yproto loss acc
global XValidation YValidation

numclass = 2; %length(unique(YValidation));

nposn = repmat(YTrain,[1 numclass]);
nposn = double(nposn);
nvaln = repmat(YValidation,[1 numclass]);
nvaln = double(nvaln);

layers = resnet_customKm_snr(size(XTrain,1:3), numclass);

%% training

miniBatchSize  = 1024;
validationFrequency = floor(numel(YTrain)/miniBatchSize);
options = trainingOptions('sgdm', ...
    'MiniBatchSize',miniBatchSize, ...
    'MaxEpochs',10, ...
    'InitialLearnRate',0.01,...
    'LearnRateSchedule','piecewise',...
    'LearnRateDropPeriod',10,...
    'ValidationFrequency',validationFrequency,...
    'ValidationData',{XValidation,nvaln},...
    'LearnRateDropFactor',0.1,...
    'Shuffle','every-epoch', ...
    'OutputFcn',@(info)proto_output(info), ...  
    'Verbose',false);

global handles
acc = 0;
handles.fig=figure(1);
handles.pb2=uicontrol('style','pushbutton','position',...
    [ 0 0 80 40],'callback',@stop_cb,'string','Stop Training');
handles.run=1;
guidata(handles.fig,handles)
subplot(2,3,1)
title(['acc:', num2str(acc)])
subplot(2,3,4)
grid on
loss = 1;
title(['loss: ',num2str(loss)]);
subplot(2,3,[ 2 3 5 6 ])
title('2d space')
g = gcf;
gcf.Color = [ 1 1 1 ];
grid on

[ net info ] = trainNetwork(XTrain,nposn,layers,options);
save('finalnet.mat','net')

%% split the data using the trained network

% can load full data here after the network is trained (full)
% or for testing use 15% dataset (even)
% load deepsig_2016_full_noamsb.mat
load deepsig_2016_even_noamsb.mat
load finalnet

%%%%%%%%%%%%%%%%%
% predict on the trained split net (Training set)
ypred = predict(net,XTrain);
X = [ypred(:,1), ypred(:,2)];

% split %%%%%%%%%%%%%%%%%%%%%%%%
load deepsig_2016_even_noamsb.mat
% load deepsig_2016_full_noamsb.mat

ypred = predict(net,XTrain);
X = [ypred(:,1), ypred(:,2)];
opts = statset('Display','off');
[idx,C] = kmeans(X,2,'Distance','sqeuclidean',...
    'Replicates',10,'Options',opts);
idxtrain = idx;

ypred = predict(net,XValidation);
X = [ypred(:,1), ypred(:,2)];
opts = statset('Display','off');
[idx,C] = kmeans(X,2,'Distance','sqeuclidean',...
    'Replicates',10,'Options',opts);
idxval = idx;

xv = XValidation;
yv = YValidation;
sv = SValidation;

xt = XTrain;
yt = YTrain;
st = STrain;

XValidation = xv(:,:,:,idxval==1);
YValidation = yv(idxval==1);
SValidation = sv(idxval==1);

XTrain = xt(:,:,:,idxtrain==1);
YTrain = yt(idxtrain==1);
STrain = st(idxtrain==1);

save('deepsig_2016_all_split1.mat','XTrain','YTrain',...
    'XValidation','YValidation','STrain','SValidation')

XValidation = xv(:,:,:,idxval==2);
YValidation = yv(idxval==2);
SValidation = sv(idxval==2);

XTrain = xt(:,:,:,idxtrain==2);
YTrain = yt(idxtrain==2);
STrain = st(idxtrain==2);

save('deepsig_2016_all_split2.mat','XTrain','YTrain',...
    'XValidation','YValidation','STrain','SValidation')


%% view histograms of the split

load deepsig_2016_all_split1.mat
subplot(1,2,1)
histogram(STrain)
cor = [2067	1954	2009	1990	1981	2015	1937	1964	1989	2027	1999	1977	2041	2014	2035	1978	2069	1991	1943	1995 ];
per = [];
count = 1;
for k = -20:2:18
    idx = SValidation == k;
    per = [per sum(idx)/cor(count) ];
    count = count + 1;
end
per1 = per;
load deepsig_2016_all_split2.mat
subplot(1,2,2)
histogram(STrain)
cor = [2067	1954	2009	1990	1981	2015	1937	1964	1989	2027	1999	1977	2041	2014	2035	1978	2069	1991	1943	1995 ];
per = [];
count = 1;
for k = -20:2:18
    idx = SValidation == k;
    per = [per sum(idx)/cor(count) ];
    count = count + 1;
end
per = [ per; per1 ]; sum(per,2), mean(per,2)

obj = 0.50 - abs(0.50 - per);
sum(sum(obj))
%%%%%%%%%%%%%%%%%

%% train a net on each cluster of data
% for modulation classification

layers = resnet_custom([1 128 2],10);
load deepsig_2016_all_split1.mat
% load deepsig_2016_all_split2.mat

validationFrequency = floor( numel(YTrain) / 128 );

options = trainingOptions('adam', ...
    'InitialLearnRate',0.01,...
    'MaxEpochs',12, ...
    'LearnRateDropFactor',0.1,...
    'LearnRateSchedule','piecewise',...
    'LearnRateDropPeriod',10,...
    'ValidationData',{XValidation,YValidation}, ...
    'ValidationFrequency', validationFrequency,...
    'MiniBatchSize',128, ...
    'Verbose',false,...
    'Plots','training-progress');

[ net, info ] = trainNetwork(XTrain,YTrain,layers,options);

%% eval snrs from net split on snr

y = [];
cor = [];
dmat = [];
for k = -20:2:18

    idx = SValidation == k;
    if sum(idx) == 0
        f = 0;
        dmat = [ dmat; zeros(1,10) ];
    else

        xx = XValidation(:,:,:,idx);
        TestPred = classify(net,xx);
        cm = confusionmat(YValidation(idx),TestPred);

        dd = diag(cm)'./sum(cm,2)';
        for n = 1:length(dd)
            if isnan(dd(n))
                dd(n) = 0;
            end
        end
        dd = dd.*100; % dd is acc of each mod
        % as 8psk am-dsb bpsk cpfsk gfsk pam4 qam16 qam64 qpsk wbfm
        dmat = [ dmat; dd ];
        f = sum(dd)/length(dd);

    end
    y = [y f]; % acc avg per snr
    cor = [ cor sum(idx) ];
end

%% %%%%% Train on one network split, save dmat2 = dmat %%%%%%%%
% then train second net and run this code to concatonate the accuracies

% 1. train network using split 1 data
% 2. run dmat2 = dmat;
% 3. train second network using split 2 data
% 4. run the below code to concatonate results \/
% 5. plot both results

dmt = zeros(size(dmat));
idx1 = dmat > dmat2;
idx2 = dmat2 >= dmat;
dmt(idx1) = dmat(idx1);
dmt(idx2) = dmat2(idx2);

%% plot accuracy values for each net
% will need to combine data from both nets for full plot /\

plot(-20:2:18,dmt(:,1:5),'-o')
hold on
plot(-20:2:18,dmt(:,6:end),'-*')
stg = ["8PSK", "AM-DSB", "BPSK", "CPFKS", "GFSK", "PAM4", "QAM16", "QAM64", "QPSK", "WBFM"];
legend(stg,'Location','northwest')
ylim([0 100])
grid on
g = gcf;
g.Color = [ 1 1 1 ];
xlabel('SNR (dB)')
ylabel('Accuracy')
title('Accuracy per Mod Type vs. SNR')

%% %%%%%%%%%%% unsup snr(x) (non ML method) %%%%%%%%%%%%%%%%%%%%%%
% load dataset
% compute snr(x) for each xtrain xval
% cluster with kmeans
% divide xtrain and xval into 2 datasets

% load deepsig_2016_full_noamsb.mat
load deepsig_2016_even_noamsb.mat

s = [];
for k = 1:size(XValidation,4)
        x = XValidation(:,:,:,k);
        x = permute(x,[ 3 2 1 ]);
        x = x(1,:) + i.*x(2,:);
        s(k) = snr(abs(x));
end

plot(s,'o')

sum(s>0)
sum(s<0)

%% calculate kmeans to get 2 clusters
idx = kmeans(s',2);

%% unsup non ml method snr(x)
% plot across each snr file
% requires splitting the dataset for each snr

y = [];
ymin = [];
ymax = [];
for k = -20:2:18
    s = [];
    stg = [ 'deepsig_2016_',num2str(k),'snr.mat'];
    stg = string(stg);
    load(stg);

    for K = 1:size(XValidation,4)
        x = XValidation(:,:,:,K);
        x = permute(x,[ 3 2 1 ]);
        x = x(1,:) + i.*x(2,:);
        s(end+1) = snr(abs(x));
    end
    y(end+1) = sum(s)/size(XValidation,4);
    ymin(end+1) = min(s);
    ymax(end+1) = max(s);

end

plot(-20:2:18,y,'o')
hold on
plot(-20:2:18,ymin,'*')
plot(-20:2:18,ymax,'*')

%% view the kmeans clusters
idx = kmeans(y',2);

plot(y(idx==1),'ro')
hold on
plot(y(idx==2),'bo')