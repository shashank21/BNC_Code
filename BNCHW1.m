%BNChomework1
%% Problem1

%Define Constants
gK = 36;
gNa = 120;
LrV = -61;
KrV = -77;
NarV = 55;
gL = 0.3;
C = 1;

%Set Initial V
V = -65;

%For a given V calculate alphas and betas
alphaN = (0.01*(-65-V+10))/((exp((-65-V+10)/10))-1);
betaN = 0.125*exp((-65-V)/80);
alphaM = 0.1*(-65-V+25)/((exp((-65-V+25)/10))-1);
betaM = 4*exp((-65-V)/18);
alphaH = 0.07*exp((-65-V)/20);
betaH = 1/((exp(((-65-V)+30)/10))+1);

%Set other Initial Conditions
n = alphaN/(alphaN+betaN);
m = alphaM/(alphaM+betaM);
h = alphaH/(alphaH+betaH);

%Set iterations
nvec = zeros(10000,1);
mvec = zeros(10000,1);
hvec = zeros(10000,1);
Vvec = zeros(10000,1);
Ivec = zeros(10000,1);
aNvec = zeros(10000,1);
bNvec = zeros(10000,1);
aMvec = zeros(10000,1);
bMvec = zeros(10000,1);
aHvec = zeros(10000,1);
bHvec = zeros(10000,1);

I_cap = zeros(10000,1);
I_K = zeros(10000,1);
I_Na = zeros(10000,1);
I_leak = zeros(10000,1);

nvec(1) = n;
mvec(1) = m;
hvec(1) = h;
Vvec(1) = V;
Ivec(1:1000) = 20; %Set injective current on for 1ms
aNvec(1) = alphaN;
bNvec(1) = betaN;
aMvec(1) = alphaM;
bMvec(1) = betaM;
aHvec(1) = alphaH;
bHvec(1) = betaH;

for i = 2:10000
    Vvec(i) = Vvec(i-1) + 0.001*((Ivec(i-1) - (gK*(nvec(i-1)^4)*((Vvec(i-1))-KrV)) - (gNa*(mvec(i-1)^3)*hvec(i-1)*((Vvec(i-1))-NarV)) - (gL*((Vvec(i-1))-LrV)))/C);
    I_cap(i-1) = (Vvec(i)-Vvec(i-1))*C/0.001;
    I_K(i-1) = (gK*(nvec(i-1)^4)*((Vvec(i-1))-KrV));
    I_Na(i-1) = (gNa*(mvec(i-1)^3)*hvec(i-1)*((Vvec(i-1))-NarV));
    I_leak(i-1) = (gL*((Vvec(i-1))-LrV))/C;
    
    %Now I need to update my rate constants. 
    aNvec(i) = (0.01*((-65-Vvec(i))+10))/((exp(((-65-Vvec(i))+10)/10))-1);
    bNvec(i) = 0.125*exp((-65-Vvec(i))/80);
    aMvec(i) = 0.1*((-65-Vvec(i)+25)/((exp((-65-Vvec(i)+25)/10))-1));
    bMvec(i) = 4*exp((-65-Vvec(i))/18);
    aHvec(i) = 0.07*exp((-65-Vvec(i))/20);
    bHvec(i) = 1/((exp(((-65-Vvec(i))+30)/10))+1);
    
    %Calculate new activation values
    nvec(i) = nvec(i-1) + 0.001*((aNvec(i)*(1-nvec(i-1)))-(bNvec(i)*nvec(i-1)));
    mvec(i) = mvec(i-1) + 0.001*((aMvec(i)*(1-mvec(i-1)))-(bMvec(i)*mvec(i-1)));
    hvec(i) = hvec(i-1) + 0.001*((aHvec(i)*(1-hvec(i-1)))-(bHvec(i)*hvec(i-1)));
        
end

%Part(i)
t = linspace(0,10,10000);
figure;
plot(t,nvec,t,mvec,t,hvec)
legend({'n','m','h'})
title('Evolution of m,n,h during an AP')
xlabel('Time (ms)')
ylabel('HH Variables')

%part(ii)
Cond_K = gK*nvec.^4;
Cond_Na = gNa*hvec.*mvec.^3;
figure;
plot(t, Cond_Na, t, Cond_K)
legend({'Na Conductance','K Conductance'})
xlabel('Time (ms)')
ylabel('Conductances (mS/cm^2)')
title('Evolution of Ion Conductances during an AP')

%part(iii)
figure;
plot(t, I_cap, t, I_leak, t, I_Na, t, I_K)
title('Evolution of Currents')
legend({'Capacitive', 'Leak', 'Na', 'K'})
xlabel('Time (ms)')
ylabel('Current (uA/cm^2)')

%part(iv)
figure;
plot(t, Vvec)
title('Action Potential')
xlabel('Time (ms)')
ylabel('Voltage (mV)')



%% Q1 Part (v)

%Aim here is to fit aNvec with a quadratic equation, and replace it in the
%above code to check model

%Note, fitting over [-100,0] was key. This didn't work over [-100,100] and
%when Vset was -65-Vset
Vset = linspace(-100, 0, 100);
aNset = (0.01*(Vset+10))./((exp((Vset+10)/10))-1);
coeff = polyfit(Vset, aNset, 2);
aNquad = coeff(1)*Vset.^2 + coeff(2)*Vset + coeff(3);

%Compare Fit
figure;
plot(Vset, aNset, Vset, aNquad)
legend({'Exp Fit', 'Quad Fit'})
title('Comparison of Exponential vs Quadratic Fit')
xlabel('V (mV)')
ylabel('aN Values')

%Redo HH model with new equation
%Define Constants
gK = 36;
gNa = 120;
LrV = -61;
KrV = -77;
NarV = 55;
gL = 0.3;
C = 1;

%Set Initial V
V = -65;

%For a given V calculate alphas and betas
alphaN = coeff(1)*(-65-V)^2 + coeff(2)*(-65-V) + coeff(3);
%Silencing old exponential intiation
%alphaN = (0.01*(-65-V+10))/((exp((-65-V+10)/10))-1);
betaN = 0.125*exp((-65-V)/80);
alphaM = 0.1*(-65-V+25)/((exp((-65-V+25)/10))-1);
betaM = 4*exp((-65-V)/18);
alphaH = 0.07*exp((-65-V)/20);
betaH = 1/((exp(((-65-V)+30)/10))+1);

%Set other Initial Conditions
n = alphaN/(alphaN+betaN);
m = alphaM/(alphaM+betaM);
h = alphaH/(alphaH+betaH);

%Set iterations
nvec = zeros(10000,1);
mvec = zeros(10000,1);
hvec = zeros(10000,1);
Vvec = zeros(10000,1);
Ivec = zeros(10000,1);
aNvec = zeros(10000,1);
bNvec = zeros(10000,1);
aMvec = zeros(10000,1);
bMvec = zeros(10000,1);
aHvec = zeros(10000,1);
bHvec = zeros(10000,1);

I_cap = zeros(10000,1);
I_K = zeros(10000,1);
I_Na = zeros(10000,1);
I_leak = zeros(10000,1);

nvec(1) = n;
mvec(1) = m;
hvec(1) = h;
Vvec(1) = V;
Ivec(1:1000) = 20; %Set injective current on for 1ms
aNvec(1) = alphaN;
bNvec(1) = betaN;
aMvec(1) = alphaM;
bMvec(1) = betaM;
aHvec(1) = alphaH;
bHvec(1) = betaH;

for i = 2:10000
    Vvec(i) = Vvec(i-1) + 0.001*((Ivec(i-1) - (gK*(nvec(i-1)^4)*((Vvec(i-1))-KrV)) - (gNa*(mvec(i-1)^3)*hvec(i-1)*((Vvec(i-1))-NarV)) - (gL*((Vvec(i-1))-LrV)))/C);
    I_cap(i-1) = (Vvec(i)-Vvec(i-1))*C/0.001;
    I_K(i-1) = (gK*(nvec(i-1)^4)*((Vvec(i-1))-KrV));
    I_Na(i-1) = (gNa*(mvec(i-1)^3)*hvec(i-1)*((Vvec(i-1))-NarV));
    I_leak(i-1) = (gL*((Vvec(i-1))-LrV))/C;
    
    %Now I need to update my rate constants. 
    aNvec(i) = coeff(1)*(-65-Vvec(i))^2 + coeff(2)*(-65-Vvec(i)) + coeff(3);
    %Silencing old exponential fit
    %aNvec(i) = (0.01*((-65-Vvec(i))+10))/((exp(((-65-Vvec(i))+10)/10))-1);
    bNvec(i) = 0.125*exp((-65-Vvec(i))/80);
    aMvec(i) = 0.1*((-65-Vvec(i)+25)/((exp((-65-Vvec(i)+25)/10))-1));
    bMvec(i) = 4*exp((-65-Vvec(i))/18);
    aHvec(i) = 0.07*exp((-65-Vvec(i))/20);
    bHvec(i) = 1/((exp(((-65-Vvec(i))+30)/10))+1);
    
    %Calculate new activation values
    nvec(i) = nvec(i-1) + 0.001*((aNvec(i)*(1-nvec(i-1)))-(bNvec(i)*nvec(i-1)));
    mvec(i) = mvec(i-1) + 0.001*((aMvec(i)*(1-mvec(i-1)))-(bMvec(i)*mvec(i-1)));
    hvec(i) = hvec(i-1) + 0.001*((aHvec(i)*(1-hvec(i-1)))-(bHvec(i)*hvec(i-1)));
        
end

%Part(i)
t = linspace(0,10,10000);
figure;
plot(t,nvec,t,mvec,t,hvec)
legend({'n','m','h'})
title('Evolution of m,n,h during an AP')
xlabel('Time (ms)')
ylabel('HH Variables')

%part(ii)
Cond_K = gK*nvec.^4;
Cond_Na = gNa*hvec.*mvec.^3;
figure;
plot(t, Cond_Na, t, Cond_K)
legend({'Na Conductance','K Conductance'})
xlabel('Time (ms)')
ylabel('Conductances (mS/cm^2)')
title('Evolution of Ion Conductances during an AP')

%part(iii)
figure;
plot(t, I_cap, t, I_leak, t, I_Na, t, I_K)
title('Evolution of Currents')
legend({'Capacitive', 'Leak', 'Na', 'K'})
xlabel('Time (ms)')
ylabel('Current (uA/cm^2)')

%part(iv)
figure;
plot(t, Vvec)
title('Action Potential')
xlabel('Time (ms)')
ylabel('Voltage (mV)')

%% Problem1b

%Aim here is to implement the reduced HH model, which will involve fewer
%DEs

%Define Constants
gK = 36;
gNa = 120;
LrV = -61;
KrV = -77;
NarV = 55;
gL = 0.3;
C = 1;

%Set Initial V
V = -65;

%%For a given V calculate alphas and betas
alphaN = (0.01*(-65-V+10))/((exp((-65-V+10)/10))-1);
betaN = 0.125*exp((-65-V)/80);
alphaM = 0.1*(-65-V+25)/((exp((-65-V+25)/10))-1);
betaM = 4*exp((-65-V)/18);

%Set other Initial Conditions
n = alphaN/(alphaN+betaN);
m = alphaM/(alphaM+betaM);

%Set iterations
nvec = zeros(10000,1);
mvec = zeros(10000,1);
Vvec = zeros(10000,1);
Ivec = zeros(10000,1);
aNvec = zeros(10000,1);
bNvec = zeros(10000,1);
aMvec = zeros(10000,1);
bMvec = zeros(10000,1);

I_cap = zeros(10000,1);
I_K = zeros(10000,1);
I_Na = zeros(10000,1);
I_leak = zeros(10000,1);

nvec(1) = n;
mvec(1) = m;
Vvec(1) = V;
Ivec(1:1000) = 20; %Set injective current on for 1ms
aNvec(1) = alphaN;
bNvec(1) = betaN;
aMvec(1) = alphaM;
bMvec(1) = betaM;

for i = 2:10000
    Vvec(i) = Vvec(i-1) + 0.001*((Ivec(i-1) - (gK*(nvec(i-1)^4)*((Vvec(i-1))-KrV)) - (gNa*(mvec(i-1)^3)*(0.89-(1.1*nvec(i-1)))*((Vvec(i-1))-NarV)) - (gL*((Vvec(i-1))-LrV)))/C);
    I_cap(i-1) = (Vvec(i)-Vvec(i-1))*C/0.001;
    I_K(i-1) = (gK*(nvec(i-1)^4)*((Vvec(i-1))-KrV));
    I_Na(i-1) = (gNa*(mvec(i-1)^3)*(0.89-(1.1*nvec(i-1)))*((Vvec(i-1))-NarV));
    I_leak(i-1) = (gL*((Vvec(i-1))-LrV))/C;
    
    %Now I need to update my rate constants. 
    aNvec(i) = (0.01*((-65-Vvec(i))+10))/((exp(((-65-Vvec(i))+10)/10))-1);
    bNvec(i) = 0.125*exp((-65-Vvec(i))/80);
    aMvec(i) = 0.1*((-65-Vvec(i)+25)/((exp((-65-Vvec(i)+25)/10))-1));
    bMvec(i) = 4*exp((-65-Vvec(i))/18);
    
    %Calculate new activation values
    nvec(i) = nvec(i-1) + 0.001*((aNvec(i)*(1-nvec(i-1)))-(bNvec(i)*nvec(i-1)));
    mvec(i) = aMvec(i)/(aMvec(i)+bMvec(i));    
end

t = linspace(0,10,10000);
figure;
plot(t,nvec,t,mvec)
legend({'n','m'})
title('Evolution of m,n during an AP')
xlabel('Time (ms)')
ylabel('HH Variables')

Cond_K = gK*nvec.^4;
Cond_Na = gNa*(0.89-(1.1*nvec(i-1))).*mvec.^3;
figure;
plot(t, Cond_Na, t, Cond_K)
legend({'Na Conductance','K Conductance'})
xlabel('Time (ms)')
ylabel('Conductances (mS/cm^2)')
title('Evolution of Ion Conductances during an AP')

figure;
plot(t, I_cap, t, I_leak, t, I_Na, t, I_K)
title('Evolution of Currents')
legend({'Capacitive', 'Leak', 'Na', 'K'})
xlabel('Time (ms)')
ylabel('Current (uA/cm^2)')

figure;
plot(t, Vvec)
title('Action Potential')
xlabel('Time (ms)')
ylabel('Voltage (mV)')


%% Problem2
%Define Constants
R = 10; %MOhm
C = 1; %nF
V_thr = 5; %mV
%V_spk = 70; %mV
dt = 1; %ms

Vvec = zeros(100,1);
Ivec = zeros(100,1);
Ivec(11:60) = 1;
Vvec(1) = 0;

%need this for loop to increase Vvec according to DE when Vvec < 5, and
%spike when reached. Then set Vvec(i+1) = 0 and have the loop
%continue
for i = 2:100
    if Vvec(i-1) == 70
        Vvec(i) = 0;
    elseif Vvec(i-1)>= 5
        Vvec(i) = 70;
    else
    Vvec(i) = Vvec(i-1) + ((Ivec(i-1) - (Vvec(i-1)/R))/C);
    end
end

figure;
plot(Vvec)
xlabel('Time (ms)')
ylabel('Voltage (mV)')
title('IAF Neuron')

%Setting injection current to be a sine function from 10 to 60ms

tsinevec1 = linspace(0, 1, 1000);
Isinevec1 = sin(2*pi*tsinevec1); %1Hz frequency
Isinevec2 = sin(4*pi*tsinevec1); %2Hz frequency
Isinevec3 = sin(10*pi*tsinevec1); %5Hz frequency
Isinevec4 = sin(20*pi*tsinevec1); %10Hz frequency
Isinevec5 = sin(40*pi*tsinevec1); %20Hz frequency
Isinevec6 = sin(100*pi*tsinevec1); %50Hz frequency
Isinevec7 = sin(200*pi*tsinevec1); %100Hz frequency

Vvecsine1 = zeros(1000,0);
Vvecsine2 = zeros(1000,0);
Vvecsine3 = zeros(1000,0);
Vvecsine4 = zeros(1000,0);
Vvecsine5 = zeros(1000,0);
Vvecsine6 = zeros(1000,0);
Vvecsine7 = zeros(1000,0);

Vvecsine1(1) = 0;
Vvecsine2(1) = 0;
Vvecsine3(1) = 0;
Vvecsine4(1) = 0;
Vvecsine5(1) = 0;
Vvecsine6(1) = 0;
Vvecsine7(1) = 0;

%Create 7 counters to keep track of number of spikes
spikecount = zeros(7,1);

%Repeat above I&F loop for various frequencies
for j = 2:1000
    if Vvecsine1(j-1) == 70
        Vvecsine1(j) = 0;
        spikecount(1) = spikecount(1) + 1;
    elseif Vvecsine1(j-1)>= 5
        Vvecsine1(j) = 70;
    else
    Vvecsine1(j) = Vvecsine1(j-1) + ((Isinevec1(j-1) - (Vvecsine1(j-1)/R))/C);
    end
end

for j = 2:1000
    if Vvecsine2(j-1) == 70
        Vvecsine2(j) = 0;
        spikecount(2) = spikecount(2) + 1;
    elseif Vvecsine2(j-1)>= 5
        Vvecsine2(j) = 70;
    else
    Vvecsine2(j) = Vvecsine2(j-1) + ((Isinevec2(j-1) - (Vvecsine2(j-1)/R))/C);
    end
end

for j = 2:1000
    if Vvecsine3(j-1) == 70
        Vvecsine3(j) = 0;
        spikecount(3) = spikecount(3) + 1;
    elseif Vvecsine3(j-1)>= 5
        Vvecsine3(j) = 70;
        
    else
    Vvecsine3(j) = Vvecsine3(j-1) + ((Isinevec3(j-1) - (Vvecsine3(j-1)/R))/C);
    end
end

for j = 2:1000
    if Vvecsine4(j-1) == 70
        Vvecsine4(j) = 0;
        spikecount(4) = spikecount(4) + 1;
    elseif Vvecsine4(j-1)>= 5
        Vvecsine4(j) = 70;
        
    else
    Vvecsine4(j) = Vvecsine4(j-1) + ((Isinevec4(j-1) - (Vvecsine4(j-1)/R))/C);
    end
end

for j = 2:1000
    if Vvecsine5(j-1) == 70
        Vvecsine5(j) = 0;
        spikecount(5) = spikecount(5) + 1;
    elseif Vvecsine5(j-1)>= 5
        Vvecsine5(j) = 70;
        
    else
    Vvecsine5(j) = Vvecsine5(j-1) + ((Isinevec5(j-1) - (Vvecsine5(j-1)/R))/C);
    end
end

for j = 2:1000
    if Vvecsine6(j-1) == 70
        Vvecsine6(j) = 0;
        spikecount(6) = spikecount(6) + 1;
    elseif Vvecsine6(j-1)>= 5
        Vvecsine6(j) = 70;
        
    else
    Vvecsine6(j) = Vvecsine6(j-1) + ((Isinevec6(j-1) - (Vvecsine6(j-1)/R))/C);
    end
end

for j = 2:1000
    if Vvecsine7(j-1) == 70
        Vvecsine7(j) = 0;
        spikecount(7) = spikecount(7) + 1;
    elseif Vvecsine7(j-1)>= 5
        Vvecsine7(j) = 70;
        
    else
    Vvecsine7(j) = Vvecsine7(j-1) + ((Isinevec7(j-1) - (Vvecsine7(j-1)/R))/C);
    end
end

figure;
x_spike = [1,2,5,10,20,50,100];
plot(x_spike, spikecount);

figure;
plot(tsinevec1, Vvecsine1)
xlabel('Time (ms)')
title('1 Hz IAF Response')
yyaxis left
ylabel('Voltage (mV)')
hold on
yyaxis right
ylabel('Current (nA)')
plot(tsinevec1, Isinevec1)
hold off

figure;
plot(tsinevec1, Vvecsine2)
xlabel('Time (ms)')
title('2 Hz IAF Response')
yyaxis left
ylabel('Voltage (mV)')
hold on
yyaxis right
ylabel('Current (nA)')
plot(tsinevec1, Isinevec2)
hold off

figure;
plot(tsinevec1, Vvecsine3)
xlabel('Time (ms)')
title('5 Hz IAF Response')
yyaxis left
ylabel('Voltage (mV)')
hold on
yyaxis right
ylabel('Current (nA)')
plot(tsinevec1, Isinevec3)
hold off

figure;
plot(tsinevec1, Vvecsine4)
xlabel('Time (ms)')
title('10 Hz IAF Response')
yyaxis left
ylabel('Voltage (mV)')
hold on
yyaxis right
ylabel('Current (nA)')
plot(tsinevec1, Isinevec4)
hold off

figure;
plot(tsinevec1, Vvecsine5)
xlabel('Time (ms)')
title('20 Hz IAF Response')
yyaxis left
ylabel('Voltage (mV)')
hold on
yyaxis right
ylabel('Current (nA)')
plot(tsinevec1, Isinevec5)
hold off

figure;
plot(tsinevec1, Vvecsine6)
xlabel('Time (ms)')
title('50 Hz IAF Response')
yyaxis left
ylabel('Voltage (mV)')
hold on
yyaxis right
ylabel('Current (nA)')
plot(tsinevec1, Isinevec6)
hold off

figure;
plot(tsinevec1, Vvecsine7)
xlabel('Time (ms)')
title('100 Hz IAF Response')
yyaxis left
ylabel('Voltage (mV)')
hold on
yyaxis right
ylabel('Current (nA)')
plot(tsinevec1, Isinevec7)
hold off

%% Problem2b

%Initializing constants and vectors
vvec_2b = zeros(1000,1);
ivec_2b = zeros(1000,1);
uvec_2b = zeros(1000,1);
a = 0.02;
b = 0.2;
c = -65;
d = 8;
vvec_2b(1) = -65;
uvec_2b(1) = b*vvec_2b(1);

%Set I impulse
ivec_2b(11:60) = 10; %From the paper

%Running Model, with dt=1 so not in equation
for i = 2:1000
    if vvec_2b(i-1) >= 30
        vvec_2b(i) = c;
        uvec_2b(i) = uvec_2b(i-1) + d;
    else
    vvec_2b(i) = vvec_2b(i-1) + 0.04*(vvec_2b(i-1)^2) + 5*vvec_2b(i-1) + 140 - uvec_2b(i-1) + ivec_2b(i-1);
    uvec_2b(i) = uvec_2b(i-1) + a*(b*vvec_2b(i-1) - uvec_2b(i-1));
        if vvec_2b(i) > 30
        vvec_2b(i) = 30;
        end
    end
end

tvec_2b = 1:100;
figure;
plot(tvec_2b, vvec_2b(1:100), tvec_2b, uvec_2b(1:100))
xlabel('Time (ms)')
title('Voltage vs Time with New Neuron Model')
ylabel('Voltage (mV)')
legend('V(t)','U(t)')

%Repeat Above with sine input of various frequencies
%Setting injection current to be a sine function through the entire 1s

tsinevec1 = linspace(0, 1, 1000);
Isinevec1 = 10*sin(2*pi*tsinevec1); %1Hz frequency
Isinevec2 = 10*sin(4*pi*tsinevec1); %2Hz frequency
Isinevec3 = 10*sin(10*pi*tsinevec1); %5Hz frequency
Isinevec4 = 10*sin(20*pi*tsinevec1); %10Hz frequency
Isinevec5 = 10*sin(40*pi*tsinevec1); %20Hz frequency
Isinevec6 = 10*sin(100*pi*tsinevec1); %50Hz frequency
Isinevec7 = 10*sin(200*pi*tsinevec1); %100Hz frequency

Vvecsine1 = zeros(1000,0);
Vvecsine2 = zeros(1000,0);
Vvecsine3 = zeros(1000,0);
Vvecsine4 = zeros(1000,0);
Vvecsine5 = zeros(1000,0);
Vvecsine6 = zeros(1000,0);
Vvecsine7 = zeros(1000,0);

Vvecsine1(1) = -65;
Vvecsine2(1) = -65;
Vvecsine3(1) = -65;
Vvecsine4(1) = -65;
Vvecsine5(1) = -65;
Vvecsine6(1) = -65;
Vvecsine7(1) = -65;

Uvecsine1 = zeros(1000,0);
Uvecsine2 = zeros(1000,0);
Uvecsine3 = zeros(1000,0);
Uvecsine4 = zeros(1000,0);
Uvecsine5 = zeros(1000,0);
Uvecsine6 = zeros(1000,0);
Uvecsine7 = zeros(1000,0);

Uvecsine1(1) = -65*b;
Uvecsine2(1) = -65*b;
Uvecsine3(1) = -65*b;
Uvecsine4(1) = -65*b;
Uvecsine5(1) = -65*b;
Uvecsine6(1) = -65*b;
Uvecsine7(1) = -65*b;

%Create 7 counters to keep track of number of spikes
spikecount = zeros(7,1);

for i = 2:1000
    if Vvecsine1(i-1) >= 30
        Vvecsine1(i) = c;
        Uvecsine1(i) = Uvecsine1(i-1) + d;
        spikecount(1) = spikecount(1) + 1;
    else
    Vvecsine1(i) = Vvecsine1(i-1) + 0.04*(Vvecsine1(i-1)^2) + 5*Vvecsine1(i-1) + 140 - Uvecsine1(i-1) + Isinevec1(i-1);
    Uvecsine1(i) = Uvecsine1(i-1) + a*(b*Vvecsine1(i-1) - Uvecsine1(i-1));
        if Vvecsine1(i) > 30
        Vvecsine1(i) = 30;
        end
    end
end

for i = 2:1000
    if Vvecsine2(i-1) >= 30
        Vvecsine2(i) = c;
        Uvecsine2(i) = Uvecsine2(i-1) + d;
        spikecount(2) = spikecount(2) + 1;
    else
    Vvecsine2(i) = Vvecsine2(i-1) + 0.04*(Vvecsine2(i-1)^2) + 5*Vvecsine2(i-1) + 140 - Uvecsine2(i-1) + Isinevec2(i-1);
    Uvecsine2(i) = Uvecsine2(i-1) + a*(b*Vvecsine2(i-1) - Uvecsine2(i-1));
        if Vvecsine2(i) > 30
        Vvecsine2(i) = 30;
        end
    end
end

for i = 2:1000
    if Vvecsine3(i-1) >= 30
        Vvecsine3(i) = c;
        Uvecsine3(i) = Uvecsine3(i-1) + d;
        spikecount(3) = spikecount(3) + 1;
    else
    Vvecsine3(i) = Vvecsine3(i-1) + 0.04*(Vvecsine3(i-1)^2) + 5*Vvecsine3(i-1) + 140 - Uvecsine3(i-1) + Isinevec3(i-1);
    Uvecsine3(i) = Uvecsine3(i-1) + a*(b*Vvecsine3(i-1) - Uvecsine3(i-1));
        if Vvecsine3(i) > 30
        Vvecsine3(i) = 30;
        end
    end
end

for i = 2:1000
    if Vvecsine4(i-1) >= 30
        Vvecsine4(i) = c;
        Uvecsine4(i) = Uvecsine4(i-1) + d;
        spikecount(4) = spikecount(4) + 1;
    else
    Vvecsine4(i) = Vvecsine4(i-1) + 0.04*(Vvecsine4(i-1)^2) + 5*Vvecsine4(i-1) + 140 - Uvecsine4(i-1) + Isinevec4(i-1);
    Uvecsine4(i) = Uvecsine4(i-1) + a*(b*Vvecsine4(i-1) - Uvecsine4(i-1));
        if Vvecsine4(i) > 30
        Vvecsine4(i) = 30;
        end
    end
end

for i = 2:1000
    if Vvecsine5(i-1) >= 30
        Vvecsine5(i) = c;
        Uvecsine5(i) = Uvecsine5(i-1) + d;
        spikecount(5) = spikecount(5) + 1;
    else
    Vvecsine5(i) = Vvecsine5(i-1) + 0.04*(Vvecsine5(i-1)^2) + 5*Vvecsine5(i-1) + 140 - Uvecsine5(i-1) + Isinevec5(i-1);
    Uvecsine5(i) = Uvecsine5(i-1) + a*(b*Vvecsine5(i-1) - Uvecsine5(i-1));
        if Vvecsine5(i) > 30
        Vvecsine5(i) = 30;
        end
    end
end

for i = 2:1000
    if Vvecsine6(i-1) >= 30
        Vvecsine6(i) = c;
        Uvecsine6(i) = Uvecsine6(i-1) + d;
        spikecount(6) = spikecount(6) + 1;
    else
    Vvecsine6(i) = Vvecsine6(i-1) + 0.04*(Vvecsine6(i-1)^2) + 5*Vvecsine6(i-1) + 140 - Uvecsine6(i-1) + Isinevec6(i-1);
    Uvecsine6(i) = Uvecsine6(i-1) + a*(b*Vvecsine6(i-1) - Uvecsine6(i-1));
        if Vvecsine6(i) > 30
        Vvecsine6(i) = 30;
        end
    end
end

for i = 2:1000
    if Vvecsine7(i-1) >= 30
        Vvecsine7(i) = c;
        Uvecsine7(i) = Uvecsine7(i-1) + d;
        spikecount(7) = spikecount(7) + 1;
    else
    Vvecsine7(i) = Vvecsine7(i-1) + 0.04*(Vvecsine7(i-1)^2) + 5*Vvecsine7(i-1) + 140 - Uvecsine7(i-1) + Isinevec7(i-1);
    Uvecsine7(i) = Uvecsine7(i-1) + a*(b*Vvecsine7(i-1) - Uvecsine7(i-1));
        if Vvecsine7(i) > 30
        Vvecsine7(i) = 30;
        end
    end
end

frq = [1, 2, 5, 10, 20, 50, 100];
figure;
plot(frq, spikecount)
xlabel('Frequency (Hz)')
ylabel('Number of Spikes')
title('Spike Count vs Frequency')

figure;
plot(tsinevec1, Vvecsine1, tsinevec1, Uvecsine1)
xlabel('Time (ms)')
title('1 Hz Response')
yyaxis left
ylabel('Voltage (mV)')
hold on
yyaxis right
ylabel('Current (nA)')
plot(tsinevec1, Isinevec1, '--')
legend('V','U','I')
hold off

figure;
plot(tsinevec1, Vvecsine2, tsinevec1, Uvecsine2)
xlabel('Time (ms)')
title('2 Hz Response')
yyaxis left
ylabel('Voltage (mV)')
hold on
yyaxis right
ylabel('Current (nA)')
plot(tsinevec1, Isinevec2, '--')
legend('V','U','I')
hold off

figure;
plot(tsinevec1, Vvecsine3, tsinevec1, Uvecsine3)
xlabel('Time (ms)')
title('5 Hz Response')
yyaxis left
ylabel('Voltage (mV)')
hold on
yyaxis right
ylabel('Current (nA)')
plot(tsinevec1, Isinevec3, '--')
legend('V','U','I')
hold off

figure;
plot(tsinevec1, Vvecsine4, tsinevec1, Uvecsine4)
xlabel('Time (ms)')
title('10 Hz Response')
yyaxis left
ylabel('Voltage (mV)')
hold on
yyaxis right
ylabel('Current (nA)')
plot(tsinevec1, Isinevec4, '--')
legend('V','U','I')
hold off

figure;
plot(tsinevec1, Vvecsine5, tsinevec1, Uvecsine5)
xlabel('Time (ms)')
title('20 Hz Response')
yyaxis left
ylabel('Voltage (mV)')
hold on
yyaxis right
ylabel('Current (nA)')
plot(tsinevec1, Isinevec5, '--')
legend('V','U','I')
hold off

figure;
plot(tsinevec1, Vvecsine6, tsinevec1, Uvecsine6)
xlabel('Time (ms)')
title('50 Hz Response')
yyaxis left
ylabel('Voltage (mV)')
hold on
yyaxis right
ylabel('Current (nA)')
plot(tsinevec1, Isinevec6, '--')
legend('V','U','I')
hold off

figure;
plot(tsinevec1, Vvecsine7, tsinevec1, Uvecsine7)
xlabel('Time (ms)')
title('100 Hz Response')
yyaxis left
ylabel('Voltage (mV)')
hold on
yyaxis right
ylabel('Current (nA)')
plot(tsinevec1, Isinevec7, '--')
legend('V','U','I')
hold off

%% Problem2c

%Setting Constants
C = 1;
R = 10;
V_rest = 0;
V_spk = 70;
tau_thresh = 50;
E_inh = -15;
E_syn= -15;
tau_syn = 15;
g_peak = 0.1;
T_max = 1500;
dT = 1;

%Initializing Vectors
V_n1 = zeros(1500,1);
V_n1(1) = E_inh;
V_n2 = zeros(1500,1);
V_n2(1) = E_inh;
U_n = zeros(1500,2);
theta_n1 = zeros(1500,1);
theta_n2 = zeros(1500,1);
z_n1 = zeros(1500,1);
z_n2 = zeros(1500,1);
g_n1 = zeros(1500,1);
g_n2 = zeros(1500,1);
deriv_v = zeros(1500,1);
deriv_t = zeros(1500,1);
deriv_z = zeros(1500,1);
deriv_g = zeros(1500,1);


I_inject1 = 1.1;
I_inject2 = 0.9;


for i = 2:1500
    
%Update Equations
if V_n1(i-1) ~= V_spk
    V_n1(i) = V_n1(i-1) + (1/C)*((-V_n1(i-1)/R) - g_n1(i-1)*(V_n1(i-1)-E_syn) + I_inject1); %What is g?  
    theta_n1(i) = theta_n1(i-1) + ((-theta_n1(i-1) + V_n1(i-1))/tau_thresh);
    z_n1(i) = z_n1(i-1) + (-z_n1(i-1)/tau_syn) + ((g_peak*exp(1))/tau_syn)*U_n(i-1,2); 
    g_n1(i) = g_n1(i-1) + (-g_n1(i-1)/tau_syn) + z_n1(i-1);
end
  
if V_n2(i-1) ~= V_spk
    V_n2(i) = V_n2(i-1) + (1/C)*((-V_n2(i-1)/R) - g_n2(i-1)*(V_n2(i-1)-E_syn) + I_inject2); %What is g? What is I_inject?
    theta_n2(i) = theta_n2(i-1) + ((-theta_n2(i-1) + V_n2(i-1))/tau_thresh);
    z_n2(i) = z_n2(i-1) + (-z_n2(i-1)/tau_syn) + ((g_peak*exp(1))/tau_syn)*U_n(i-1,1); %What is U
    g_n2(i) = g_n2(i-1) + (-g_n2(i-1)/tau_syn) + z_n2(i-1);
end

%Reset if Spiked
if V_n1(i-1) == V_spk
    V_n1(i) = E_inh;
end
if V_n2(i-1) == V_spk
    V_n2(i) = E_inh;
end

%Compare Thresholds
if V_n1(i) >= theta_n1(i)
    V_n1(i) = V_spk;
    U_n(i,:) = [1,0];
end
if V_n2(i) >= theta_n2(i)
    V_n2(i) = V_spk;
    U_n(i,:) = [0,1];
end

end

figure;
plot(V_n1)
hold on
plot(V_n2)
hold off
xlim([-100, 1600])
ylim([-20, 80])
xlabel('Time (ms)')
ylabel('Voltage (mV)')
legend('Neuron 1', 'Neuron 2')
title('Two-Neuron Oscillator')


%% Problem3

R_inp = dlmread('spikes.txt');
S_inp = dlmread('Stimulus.txt');
S_mat = zeros(181,20);

%Creates a mask from 0-20000ms with each 1 representing stimulus presence
%in that 100ms bin
mask_mat = zeros(200,1);
for i = 1:23
   val_low = S_inp(i,1);
   val_high = S_inp(i,2);
   R_L = floor(val_low/100);
   R_H = floor(val_high/100);
   mask_mat((R_L)+1) = 1;
   mask_mat((R_H)+1) = 1;
end

%Creates the stimulus matrix with 2s rows in 100ms bins (so 20 columns)
for j = 1:181
    S_mat(j,:) = mask_mat(j:j+19);
end

%Breaks up R_inp from 'spikes.txt' into 20s  time bins

timeset1 = R_inp<20; %Instead of find(R_inp<20) this creates 0s and 1s, with 1s where R_inp<20
ts1 = R_inp(timeset1); %Outputs vector with all of the above spike times
timeset2 = R_inp>=20 & R_inp<40;
ts2 = R_inp(timeset2);
timeset3 = R_inp>=40 & R_inp<60;
ts3 = R_inp(timeset3);
timeset4 = R_inp>=60 & R_inp<80;
ts4 = R_inp(timeset4);
timeset5 = R_inp>=80 & R_inp<100;
ts5 = R_inp(timeset5);

%In each response vector I need to ignore first 2 seconds and then 
%bin in 100ms intervals
R_1 = zeros(181,1);
R_2 = zeros(181,1);
R_3 = zeros(181,1);
R_4 = zeros(181,1);
R_5 = zeros(181,1);
for a = 1:length(ts1)
    for b = 1:181
      if ts1(a) >= 2 + (b-1)*0.1 && ts1(a) < 2 + b*0.1
          R_1(b) = R_1(b)+1;
      end
    end      
end

for a = 1:length(ts2)
    for b = 1:181
      if ts2(a) >= 22 + (b-1)*0.1 && ts2(a) < 22 + b*0.1
          R_2(b) = R_2(b)+1;
      end
    end      
end

for a = 1:length(ts3)
    for b = 1:181
      if ts3(a) >= 42 + (b-1)*0.1 && ts3(a) < 42 + b*0.1
          R_3(b) = R_3(b)+1;
      end
    end      
end

for a = 1:length(ts4)
    for b = 1:181
      if ts4(a) >= 62 + (b-1)*0.1 && ts4(a) < 62 + b*0.1
          R_4(b) = R_4(b)+1;
      end
    end      
end

for a = 1:length(ts5)
    for b = 1:181
      if ts5(a) >= 82 + (b-1)*0.1 && ts5(a) < 82 + b*0.1
          R_5(b) = R_5(b)+1;
      end
    end      
end

%Linear fit: Calculate matrix S that minimizes (R-SW)^2
%From wikipedia: "a common use of MP Pseudoinverse is to apply a least
%squares solution to a system of linear equations"

R_avg = (R_1 + R_2 + R_3 + R_4)/4;
W = pinv(S_mat)*R_avg;

%Compare fit with R_5
R_guess = S_mat*W;
x_graph = linspace(2, 20, 181); 
figure;
plot(x_graph, R_guess)
hold on
plot(x_graph, R_5)
hold off
title('Linear Fit of Spike Response to Stimulus')
xlabel('Time (s)')
ylabel('# of spikes')
legend('Predicted','Trial 5')

%Non-linear fit: paper says to "compare linear filter g(t) to actual 
%R_avg and average over bins of g(t) containing equal number of points"
%This currently does not work at all
plot(sort(R_guess), sort(R_avg))
xlabel('Predicted R spike count per bin')
ylabel('Average Spike Count over 4 trials per bin')
title('Non-Linearity (According to paper, Hz)')

%Sorts and indices my prediction and the average over 4 trials
[R_guess_ord, R_guess_ind] = sort(R_guess);
[R_avg_ord, R_avg_ind] = sort(R_avg);

%Averages across bins of 3, as done in paper, in order to simulate the
%"non
newvec_1 = zeros(60,1);
newvec_2 = zeros(60,1);
for a = 1:60
   newvec_1(a) = mean(R_avg_ord((a-1)*3+1:a*3));
   newvec_2(a) = mean(R_guess_ord((a-1)*3+1:a*3));
end
newvec_3 = (newvec_1 + newvec_2)./2;
    
new_R_guess = zeros(181,1);
for b = 1:60
    new_R_guess((b-1)*3+1:b*3) = newvec_3(b);
end
new_R_guess(181) = R_guess_ord(181);


[~, newsort] = sort(R_guess_ind);
figure;
plot(x_graph, new_R_guess(newsort))
hold on
plot(x_graph, R_5)
hold off
title('LN Fit of Spike Response to Stimulus')
ylabel('# Of Spikes')
xlabel('Time (s)')

    
%% Problem 4

%Need to run the end of problem 3 to get predicted firing rate of neuron 
%as a function of t

lambda_t = new_R_guess(newsort);
lambda_max = max(lambda_t);
ratio = lambda_t/lambda_max;
poisson_spike = zeros(181,1000);
U = zeros(181,1000);

%Replace this with numeric array later, more effective
%Creating a bunch of different spike trains
for j = 1:1000
    U(:,j) = rand([181, 1]);
    for i = 1:181
    
        if U(i,j) <= ratio(i)
        poisson_spike(i,j) = 1;
        end
    
    end
end

%Calculating ISI's across ALL spike trains

ISI_value = zeros;
index_spike = zeros;
for i = 1:1000
    index_spike = find(poisson_spike(:,i), 181);
    for j = 1:length(index_spike)-1
        ISI_value(i+j-1) = index_spike(j+1) - index_spike(j) + 1;
    end
end



        








