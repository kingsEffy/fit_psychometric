S = load('svm_sample_neural_timeSeries.mat'); % fields: traces (T x N x F), coherences, choice_right
X = S.traces;      % trials x neurons x frames
y = double(S.choice_right); 
coh = S.coherences;

N = 50;                               % try 10 or 50 neurons
ci = 2;                                % 1/2/3 for [16,40,100] if you map below
cohList = unique(coh); cVal = cohList(ci);
mask = coh == cVal;

Xc = X(mask,:,:); yc = y(mask);
idxN = randperm(size(Xc,2), N);       % choose N neurons

% split trials
idx = randperm(size(Xc,1)); nt = floor(0.25*numel(idx));
te = idx(1:nt); tr = idx(nt+1:end);

% standardise per‑neuron using training set
mu = mean(Xc(tr,idxN,:),[1 3]); sd = std(Xc(tr,idxN,:),0,[1 3])+1e-6;

acc_t = nan(size(Xc,3),1);
for f = 1:size(Xc,3)
    Xtr = squeeze((Xc(tr,idxN,f) - mu)./sd);
    Xte = squeeze((Xc(te,idxN,f) - mu)./sd);
    M   = fitcsvm(Xtr, 2*yc(tr)-1, 'KernelFunction','linear', 'BoxConstraint',1, ...
                  'Standardize', false, 'ClassNames',[-1 1]);
    yhat = predict(M, Xte);
    % acc_t(f) = mean(yhat == (2*yc(te)-1));
    yte       = 2*yc(te) - 1;   yte = yte(:);
    yhat      = yhat(:);
    acc_t(f)  = sum(yhat == yte) / numel(yte);

end

plot(linspace(0,1.5,numel(acc_t)), 100*acc_t); yline(50,'--');
xlabel('Time (s)'); ylabel('Decoder accuracy (%)');
title(sprintf('Time-resolved SVM (N=%d, coh=%d%%)', N, cVal));
