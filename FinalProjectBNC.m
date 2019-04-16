%Final Project Code

%% PCA on Different Odors across ROIs

%Calculate Average Power (Pre-Stim)

%Odor o_1o3o02
%Plot Odor Profile
plot(o_1o3o02(2,:))
figure;
plot(o_1o3o02(1,:))

%Calculate Pre-Stim/Stim/Post-Stim Average Value
o_1o3o02_prestim = zeros(922,1);
o_1o3o02_stim = zeros(922,1);
o_1o3o02_poststim = zeros(922,1);

for i = 2:923
o_1o3o02_prestim(i-1,1) = mean(o_1o3o02(i,1:40));
o_1o3o02_stim(i-1,1) = mean(o_1o3o02(i,41:56));
o_1o3o02_poststim(i-1,1) = mean(o_1o3o02(i,57:116));
end

%Odor o_1o3o04

%Calculate Pre-Stim/Stim/Post-Stim Average Value
o_1o3o04_prestim = zeros(922,1);
o_1o3o04_stim = zeros(922,1);
o_1o3o04_poststim = zeros(922,1);

for i = 2:923
o_1o3o04_prestim(i-1,1) = mean(o_1o3o04(i,1:40));
o_1o3o04_stim(i-1,1) = mean(o_1o3o04(i,41:56));
o_1o3o04_poststim(i-1,1) = mean(o_1o3o04(i,57:116));
end

%Odor o_Acet02
%Calculate Pre-Stim/Stim/Post-Stim Average Value
o_Acet02_prestim = zeros(922,1);
o_Acet02_stim = zeros(922,1);
o_Acet02_poststim = zeros(922,1);

for i = 2:923
o_Acet02_prestim(i-1,1) = mean(o_Acet02(i,1:40));
o_Acet02_stim(i-1,1) = mean(o_Acet02(i,41:56));
o_Acet02_poststim(i-1,1) = mean(o_Acet02(i,57:116));
end

%Odor o_Acet04
o_Acet04_prestim = zeros(922,1);
o_Acet04_stim = zeros(922,1);
o_Acet04_poststim = zeros(922,1);

for i = 2:923
o_Acet04_prestim(i-1,1) = mean(o_Acet04(i,1:40));
o_Acet04_stim(i-1,1) = mean(o_Acet04(i,41:56));
o_Acet04_poststim(i-1,1) = mean(o_Acet04(i,57:116));
end

%Odor o_Bzald02
o_Bzald02_prestim = zeros(922,1);
o_Bzald02_stim = zeros(922,1);
o_Bzald02_poststim = zeros(922,1);

for i = 2:923
o_Bzald02_prestim(i-1,1) = mean(o_Bzald02(i,1:40));
o_Bzald02_stim(i-1,1) = mean(o_Bzald02(i,41:56));
o_Bzald02_poststim(i-1,1) = mean(o_Bzald02(i,57:116));
end

%Odor o_Bzald04
o_Bzald04_prestim = zeros(922,1);
o_Bzald04_stim = zeros(922,1);
o_Bzald04_poststim = zeros(922,1);

for i = 2:923
o_Bzald04_prestim(i-1,1) = mean(o_Bzald04(i,1:40));
o_Bzald04_stim(i-1,1) = mean(o_Bzald04(i,41:56));
o_Bzald04_poststim(i-1,1) = mean(o_Bzald04(i,57:116));
end

%Odor o_EA02
o_EA02_prestim = zeros(922,1);
o_EA02_stim = zeros(922,1);
o_EA02_poststim = zeros(922,1);

for i = 2:923
o_EA02_prestim(i-1,1) = mean(o_EA02(i,1:40));
o_EA02_stim(i-1,1) = mean(o_EA02(i,41:56));
o_EA02_poststim(i-1,1) = mean(o_EA02(i,57:116));
end

%Odor o_EA04
o_EA04_prestim = zeros(922,1);
o_EA04_stim = zeros(922,1);
o_EA04_poststim = zeros(922,1);

for i = 2:923
o_EA04_prestim(i-1,1) = mean(o_EA04(i,1:40));
o_EA04_stim(i-1,1) = mean(o_EA04(i,41:56));
o_EA04_poststim(i-1,1) = mean(o_EA04(i,57:116));
end

%Odor o_EB02
o_EB02_prestim = zeros(922,1);
o_EB02_stim = zeros(922,1);
o_EB02_poststim = zeros(922,1);

for i = 2:923
o_EB02_prestim(i-1,1) = mean(o_EB02(i,1:40));
o_EB02_stim(i-1,1) = mean(o_EB02(i,41:56));
o_EB02_poststim(i-1,1) = mean(o_EB02(i,57:116));
end

%Odor o_EB04
o_EB04_prestim = zeros(922,1);
o_EB04_stim = zeros(922,1);
o_EB04_poststim = zeros(922,1);

for i = 2:923
o_EB04_prestim(i-1,1) = mean(o_EB04(i,1:40));
o_EB04_stim(i-1,1) = mean(o_EB04(i,41:56));
o_EB04_poststim(i-1,1) = mean(o_EB04(i,57:116));
end

%Odor o_MH02
o_MH02_prestim = zeros(922,1);
o_MH02_stim = zeros(922,1);
o_MH02_poststim = zeros(922,1);

for i = 2:923
o_MH02_prestim(i-1,1) = mean(o_MH02(i,1:40));
o_MH02_stim(i-1,1) = mean(o_MH02(i,41:56));
o_MH02_poststim(i-1,1) = mean(o_MH02(i,57:116));
end

%Plot o_MH04
o_MH04_prestim = zeros(922,1);
o_MH04_stim = zeros(922,1);
o_MH04_poststim = zeros(922,1);

for i = 2:923
o_MH04_prestim(i-1,1) = mean(o_MH04(i,1:40));
o_MH04_stim(i-1,1) = mean(o_MH04(i,41:56));
o_MH04_poststim(i-1,1) = mean(o_MH04(i,57:116));
end

%Plot o_PO
o_PO_prestim = zeros(922,1);
o_PO_stim = zeros(922,1);
o_PO_poststim = zeros(922,1);

for i = 2:923
o_PO_prestim(i-1,1) = mean(o_PO(i,1:40));
o_PO_stim(i-1,1) = mean(o_PO(i,41:56));
o_PO_poststim(i-1,1) = mean(o_PO(i,57:116));
end

%% Correlation Matrix Generation

%Input is a time x ROI matrix
%Correlation Matrix durin stim
CorrMat = paircorr(o_1o3o02(2:923,41:56)',o_1o3o02(2:923,41:56)');
figure;
imagesc(CorrMat,[-1,1])

%Correlation matrix during prestim and stim
CorrMat_2 = paircorr(o_1o3o02(2:923,1:56)',o_1o3o02(2:923,1:56)');
imagesc(CorrMat_2,[-1,1])


%correlation matrix during prestim
Corr_mat_prestim = paircorr(o_1o3o02(2:923,1:40)',o_1o3o02(2:923,1:40)');
imagesc(Corr_mat_prestim, [-1,1])

diff = CorrMat - Corr_mat_prestim;









