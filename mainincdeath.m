clc;
clear all; %#ok<CLALL>
close all;
tic
%% Message
msg='Full Runthrough - Checking Functionality';
%% Simulation Parameters
send=0;
     
width=5.5;
height=4;
N=10000;          % number of agents for the stochastic simulation

beta=0.99;       % discount factor
gamma=3;         % utility-function parameter
alpha=0.36;      % share of capital in the production function
delta=0.02;      % depreciation rate
delta_a=0.0131;    % (1-delta_a) is the productivity level in a bad state, 
                 % and (1+delta_a) is the productivity level in a good state
p_death=0.003;    % probability of dying at the end of next period   
IHT=0.05;         % get 10%leave at 0.3% level of inheritance tax
redist=1;        % Does inheritance tax revenue get redistributed 

testtrans=0;     % Is it testing transitions between states
period=50;       % How long is each test period

scale=3;                 
mu = 0.2;        % unemployment benefits as a share of wage
l_bar=1/0.93;    % time endowment; normalizes labor supply in a bad state
T=5100;          % simulation length
ndiscard=100;    % number of periods to discard

nstates_id=2;    % number of states for the idiosyncratic shock
nstates_ag=6;    % number of states for the aggregate shock

epsilon_u=0;     % idiosyncratic shock if the agent is unemployed
epsilon_e=1;     % idiosyncratic shock if the agent is employed

ur_b=0.1;        % unemployment rate in a bad aggregate state
er_b=(1-ur_b);   % employment rate in a bad aggregate state

ur_g=0.05;       % unemployment rate in a good aggregate state
er_g=(1-ur_g);   % employment rate in a good aggregate state

ur_u1=0.1;       % unemployment rate in an uncertain from bad aggregate state
er_u1=(1-ur_u1);   % employment rate in an uncertain from bad aggregate state

ur_u2=0.06;       % unemployment rate in an uncertain from good aggregate state
er_u2=(1-ur_u2);  % employment rate in an uncertain from good aggregate state

ur_b1=0.12;      % unemployment rate in a v.bad aggregate state
er_b1=(1-ur_b1); % employment rate in a v.bad aggregate state

ur_g1=0.03;      % unemployment rate in a v.good aggregate state
er_g1=(1-ur_g1); % employment rate in a v.good aggregate state
%% Stochastic beta

betas=[0.988 0.99 0.992];

pi_beta=   [0.9975      0.0025      0
            0.0003125	0.999375	0.0003125
            0           0.0025      0.9975];
        
init_dcf=[0.1 0.8 0.1];
%% Matrix of transition probabilities

prob=[0.511875	0.34125	0.03146875	0.09040625	0.015	0.01	0	0	0	0	0	0
0.037916775	0.815208225	0.003280925	0.118594075	0.001111114	0.023888886	0	0	0	0	0	0
0.07520625	0.04666875	0.289375325	0.563749675	0	0	0.014333343	0.010666657	0	0	0	0
0.008887125	0.112987875	0.02969835	0.82342665	0	0	0.000824457	0.024175543	0	0	0	0
0	0	0	0	0.42	0.28	0	0	0.102	0.048	0.0275	0.1225
0	0	0	0	0.0311112	0.6688888	0	0	0.008666686	0.141333314	0.0019496	0.1480504
0	0	0	0	0	0	0.3635336	0.3364664	0.097361538	0.052638462	0.037400057	0.112599943
0	0	0	0	0	0	0.0214448	0.6785552	0.012938	0.137062	0.002395314	0.147604686
0.11	0.09	0	0	0	0	0	0	0.495	0.305	0	0
0.007728914	0.192271086	0	0	0	0	0	0	0.041555657	0.758444343	0	0
0	0	0.096666743	0.103333257	0	0	0	0	0	0	0.175355616	0.624644384
0	0	0.007315657	0.192684343	0	0	0	0	0	0	0.019349002	0.780650998];
%% Steady State and Beliefs
kss=((1/beta-(1-delta))/alpha)^(1/(alpha-1)); 
truematrix=prob;
beliefs=prob;
%% Generation of shocks
prob=truematrix;
[idshock,agshock,deathshock,betashock]  = SHOCKSincdeath(prob,T,N,ur_b,p_death,IHT,testtrans,period...
    ,betas,pi_beta,init_dcf);
prob=beliefs;
%% Grid for capital in the individual problem (savings)

k_min=-11;                 % minimum grid-value of capital
k_max=900;                 % maximum grid-value of capital
ngridk=100;                % number of grid points
x=linspace(0,0.5,ngridk)';   
y=x.^7/max(x.^7);          % distribution of grid points
k=k_min+(k_max-k_min)*y;   

% Grid for mean of capital

km_min=30;                           % minimum grid-value of the mean of 
                                     % capital distribution, km 
km_max=60;                           % maximum grid value of km
ngridkm=5;                           % number of grid points for km 
km=linspace(km_min,km_max,ngridkm)'; 
%% Parameters of idiosyncratic and aggregate shocks

epsilon=zeros(nstates_id,1);  % vector1 of possible idiosyncratic states
epsilon(1)=epsilon_u; epsilon(2)=epsilon_e; 
epsilon2=zeros(nstates_id,1); % vector2 of possible idiosyncratic states 
epsilon2(1)=1; epsilon2(2)=2;   

a=zeros(nstates_ag,1);        % vector of possible aggregate states
a(1)=1-delta_a; a(2)=1+delta_a;
a(3)=1-4*delta_a/3; a(4)=1+2*delta_a/3;
a(5)=1-3*delta_a; a(6)=1+3*delta_a; 

a2=zeros(nstates_ag,1);       % vector of possible aggregate states
a2(1)=1; a2(2)=2; a2(3)=3;
a2(4)=4; a2(5)=5; a2(6)=6;    
%% Initial Conditions

kprime=zeros(ngridk,ngridkm,nstates_ag,nstates_id); % Initial policy function
% Initial capital function
kprime1=kprime;
for i=1:ngridkm
   for j=1:nstates_ag
      for h=1:nstates_id
         kprime1(:,i,j,h)=0.9*k;
      end
   end
end
kprime2=kprime1;
kprime3=kprime1;

% Initial distribution of capital is chosen so that aggregate capital is
% near the steady state value, kss  


% kcross=zeros(1,N)+kss;  % initial capital of all agents is equal to kss
load('dist','kcross1');
kcross=kcross1;

B=[0 1 0 1 0 1 0 1 0 1 0 1]; % starting guess for law of motion
load('values','B'); % values from previous run
%% Convergence Parameters

dif_B=10^10;    % low of motion starting difference
criter_k=1e-5;  % convergence tolerance for policy rule
criter_B=1e-5;  % convergence tolerance for law of motion
update_k=0.2;   % updating parameter for the policy rule
update_B=0.25;  % updating parameter for the law of motion
%% Solving the Model

iteration=0;      
init_time=clock; % initialize the time clock

while dif_B>criter_B 
  tic                       
[kprime,kmprime,kprime1,kprime2,kprime3,c1,c2,c3,prob_b1e,prob_b1u,prob_be,prob_bu,prob_g1e,...
    prob_g1u,prob_ge,prob_gu,prob_u1e,prob_u1u,prob_u2u,prob_u2e]=INDIVIDUALincDeath(...
    prob,ur_b,ur_g,ur_u1,ur_u2,ur_g1,ur_b1,ngridk,ngridkm,nstates_ag,...
    nstates_id,k,km,er_b,er_g,er_u1,er_u2,er_b1,er_g1,a,epsilon,l_bar,alpha,delta,gamma,betas,mu,km_max,...
    km_min,kprime1,kprime2,kprime3,B,criter_k,k_min,k_max,update_k,p_death,redist,IHT);
            % compututing a solution to the individual problem

[kmts,kcross1,~]  = AGGREGATE_death(T,idshock,agshock,betashock,km_max,km_min,kprime,km,k,...
    epsilon2,k_min,k_max,kcross,a2,N,deathshock,redist,IHT,p_death,betas);

% Time series for the regression 

ibad=0;           % count how many times the aggregate shock was bad
igood=0;          % count how many times the aggregate shock was good
iunc1=0;          % count how many times the aggregate shock was unc1
iunc2=0;          % count how many times the aggregate shock was unc2
ibad1=0;          % count how many times the aggregate shock was vbad
igood1=0;         % count how many times the aggregate shock was vgood

% Regression Variables 
xbad=0;  ybad=0;  
xunc1=0; yunc1=0;
xunc2=0; yunc2=0;
xbad1=0; ybad1=0;
xgood1=0; ygood1=0;
for i=ndiscard+1:T-1
   if agshock(i)==1
      ibad=ibad+1;
      xbad(ibad,1)=log(kmts(i)); %#ok<SAGROW>
      ybad(ibad,1)=log(kmts(i+1)); %#ok<SAGROW>
   elseif agshock(i)==2
      igood=igood+1;
      xgood(igood,1)=log(kmts(i)); %#ok<SAGROW>
      ygood(igood,1)=log(kmts(i+1)); %#ok<SAGROW>
   elseif agshock(i)==3
      iunc1=iunc1+1;
      xunc1(iunc1,1)=log(kmts(i)); %#ok<SAGROW>
      yunc1(iunc1,1)=log(kmts(i+1)); %#ok<SAGROW>
   elseif agshock(i)==4
      iunc2=iunc2+1;
      xunc2(iunc2,1)=log(kmts(i)); %#ok<SAGROW>
      yunc2(iunc2,1)=log(kmts(i+1));    %#ok<SAGROW>  
   elseif agshock(i)==5
      ibad1=ibad1+1;
      xbad1(ibad1,1)=log(kmts(i));%#ok<SAGROW>
      ybad1(ibad1,1)=log(kmts(i+1));  %#ok<SAGROW>
   else
      igood1=igood1+1;
      xgood1(igood1,1)=log(kmts(i)); %#ok<SAGROW>
      ygood1(igood1,1)=log(kmts(i+1));   %#ok<SAGROW>   
   end  
end

[B1(1:2),~,~,~,s5]=regress(ybad,[ones(ibad,1) xbad]);R2bad=s5(1); 
[B1(3:4),~,~,~,s5]=regress(ygood,[ones(igood,1) xgood]);R2good=s5(1);
[B1(5:6),~,~,~,s5]=regress(yunc1,[ones(iunc1,1) xunc1]);R2unc1=s5(1);
[B1(7:8),~,~,~,s5]=regress(yunc2,[ones(iunc2,1) xunc2]);R2unc2=s5(1);
[B1(9:10),~,~,~,s5]=regress(ybad1,[ones(ibad1,1) xbad1]);R2bad1=s5(1);
[B1(11:12),s2,s3,s4,s5]=regress(ygood1,[ones(igood1,1) xgood1]);R2good1=s5(1);
dif_B=norm(B-B1) %#ok<NOPTS> % difference in low of motion coefficients

% Ensuring in ergodic set
if dif_B>(criter_B*10) 
    kcross=kcross1; % creating a new capital distribution
end
toc
B=B1*update_B+B*(1-update_B); % update law of motion guesses
iteration=iteration+1 %#ok<NOPTS>

end
%% R2s
end_time=clock;               % end the time clock
et=etime(end_time,init_time); % compute time in seconds
disp('Elapsed Time (in seconds):'); et %#ok<NOPTS>
disp('Iterations');          iteration %#ok<NOPTS>
format long g; 
disp('R^2 bad aggregate shock:'); R2bad(1)
disp('R^2 good aggregate shock:'); R2good(1)
disp('R^2 unc1 aggregate shock:'); R2unc1(1)
disp('R^2 unc2 aggregate shock:'); R2unc2(1)
disp('R^2 bad1 aggregate shock:'); R2bad1(1)
disp('R^2 good1 aggregate shock:'); R2good1(1)
format;  
%% Ergodic Distribution
figure(3)
set(gcf,'Units','inches',...
'Position',[0 0 width height],...
'PaperPositionMode','auto','PaperOrientation','landscape','PaperUnits','Inches','PaperSize',[width, height]);
histogram(kcross1,'BinWidth',10,'Normalization','Probability')
xlabel('Wealth')
ylabel('Density')
xlim([-20 800])
set(gca,'FontUnits','points',...
'FontWeight','normal',...
'FontSize',11,...
'FontName','Times New Roman');
print -dpdf -bestfit ergwealthdist

[kmts,kcross1,~]  = AGGREGATE_death(T,idshock,agshock,betashock,km_max,km_min,kprime,km,k,...
    epsilon2,k_min,k_max,kcross,a2,N,deathshock,redist,IHT,p_death,betas);

weal=sort(kcross1);
totalwealth=sum(weal);

for i=1:N
lorenzstoc(i)=100*sum(weal(1:i))/totalwealth; %#ok<SAGROW>
end


%
% Lorenz curve data
%

cumpop=[0 5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 ...
     95 96 97 98 99 100];

 %cumpop1=cumpop*100;
cumwealth= [0.00000 -0.04400 0.05100 0.23600 0.54500 1.04900 1.82600 ...
    2.94700 4.45500 6.35600 8.70200 11.53500 14.93100 18.94800 23.73300 ...
    29.39800 36.26500 44.65600 55.17600 69.45000 73.05800 ...
    77.09200 81.71200 87.36900 100.00000];
cumwealth1=cumwealth/100;

pop=linspace(0,100,N);

cd 'C:\Users\Tim\Dropbox\MatlabCodes\sandbox\onebeta'
load Solution_to_model_new lorenz
cd 'C:\Users\Tim\Dropbox\MatlabCodes\sandbox\6statesstochasticbeta'

figure(4)
set(gcf,'Units','inches',...
'Position',[0 0 width height],...
'PaperPositionMode','auto','PaperOrientation','landscape','PaperUnits','Inches','PaperSize',[width, height]);
plot(pop,lorenz,':',pop,lorenzstoc,'--',cumpop,cumwealth,'-','LineWidth',2)
legend('Baseline','Stockastic Discount Factor','Data','location','northwest')
xlabel('Cumulative population')
ylabel('Cumulative Share of Wealth')
xlim([0 100]);
ylim([-5 100]);
set(gca,'FontUnits','points',...
'FontWeight','normal',...
'FontSize',11,...
'FontName','Times New Roman');
print -dpdf -bestfit erglorenz
%% Time Series

kmalm=zeros(T,1);  % represents aggregate capital from law of motion
kmalm(1)=kmts(1);  
                                   
for t=1:T-1       % compute kmalm
   if agshock(t)==1
      kmalm(t+1)=exp(B(1)+B(2)*log(kmalm(t)));
   elseif agshock(t)==2
      kmalm(t+1)=exp(B(3)+B(4)*log(kmalm(t)));
   elseif agshock(t)==3
      kmalm(t+1)=exp(B(5)+B(6)*log(kmalm(t)));
   elseif agshock(t)==4
      kmalm(t+1)=exp(B(7)+B(8)*log(kmalm(t)));      
   elseif agshock(t)==5
      kmalm(t+1)=exp(B(9)+B(10)*log(kmalm(t)));
   else 
       kmalm(t+1)=exp(B(11)+B(12)*log(kmalm(t)));
   end
end
figure(5)
set(gcf,'Units','inches',...
'Position',[0 0 width height],...
'PaperPositionMode','auto','PaperOrientation','landscape','PaperUnits','Inches','PaperSize',[width, height]);
Tts=1:1:T;
axis([min(Tts) max(Tts) min(kmts)*0.99 max(kmts)*1.01]); axis manual; hold on
plot (Tts,kmts(1:T,1),'-',Tts,kmalm(1:T,1),'--','LineWidth',1),xlabel('Time'), ...
    ylabel('Aggregate capital series')
legend('Implied by individual policy rule', 'Aggregate law of motion')
hold off
set(gca,'FontUnits','points',...
'FontWeight','normal',...
'FontSize',11,...
'FontName','Times New Roman');
print -dpdf -bestfit lom
%% Aggregate Panel

employment_rates=[er_b er_g er_u1 er_u2 er_b1 er_g1];

kcrosstest=horzcat(kcross,kcross,kcross);

[idshock,agshock,deathshock,betashock]  = SHOCKSpanel(prob,T,3*N,ur_b,p_death,IHT,testtrans,period...
    ,betas,pi_beta,init_dcf);
[kmts,~,~]  = AGGREGATE_death(T,idshock,agshock,betashock,km_max,km_min,kprime,km,k,...
    epsilon2,k_min,k_max,kcrosstest,a2,3*N,deathshock,redist,IHT,p_death,betas);


output=zeros(300,1);
consumption=zeros(300,1);
wage=zeros(300,1);
interestrate=zeros(300,1);
for t=2:300
    output(t,1)=a(agshock(t))*(kmts(t)^alpha)*(employment_rates(agshock(t))/l_bar)^(1-alpha);
    consumption(t,1)=kmts(t-1).*(1-delta)+output(t-1)-kmts(t);
    wage(t,1)=(1-alpha)*a(agshock(t))*(kmts(t)/(l_bar*employment_rates(agshock(t))))^alpha;
    interestrate(t,1)=alpha*a(agshock(t))*(kmts(t)/(l_bar*employment_rates(agshock(t))))^(alpha-1);
end

unemployment=2-mean(idshock,2);

range=[10 248];

figure(6)
subplot(5,1,1)
plot(output(range(1):range(2)))
title('Output')
ylim([3.2 3.8])
set(gca,'FontUnits','points',...
'FontWeight','normal',...
'FontSize',11,...
'FontName','Times New Roman');
subplot(5,1,2)
plot(consumption(range(1):range(2)))
title('Consumption')
ylim([2.55 2.75])
set(gca,'FontUnits','points',...
'FontWeight','normal',...
'FontSize',11,...
'FontName','Times New Roman');
subplot(5,1,3)
plot(wage(range(1):range(2)))
title('Wage')
ylim([2.4 2.55])
set(gca,'FontUnits','points',...
'FontWeight','normal',...
'FontSize',11,...
'FontName','Times New Roman');
subplot(5,1,4)
plot(interestrate(range(1):range(2)))
title('Interest Rate')
ylim([0.028 0.036])
set(gca,'FontUnits','points',...
'FontWeight','normal',...
'FontSize',11,...
'FontName','Times New Roman');
subplot(5,1,5)
plot(unemployment(range(1):range(2)));
title('Unemployment Rate')
ylim([0 0.15])
set(gca,'FontUnits','points',...
'FontWeight','normal',...
'FontSize',11,...
'FontName','Times New Roman');
print('Panel','-dpdf','-fillpage')
%% Individual Panel
k_ind=ones(T,1)*kss/2;
income_ind=ones(T,1);
c_ind=ones(T,1);
individual=1;
ur=ones(1,nstates_ag)-[er_b er_g er_u1 er_u2 er_b1 er_g1];

idshock(190:280,1)=2;
idshock(212:217,1)=1;
idshock(225:226,1)=1;
idshock(238:241,1)=1;


for t=1:300
    
    %if deathshock(t,individual)==1 %&& (t<20 || t>300)
    % Individual capital

    k_ind(t+1,1)=interpn(k,km,kprime(:,:,agshock(t),idshock(t,individual),2),k_ind(t,1),kmts(t),'spline');

    k_ind(t+1,1)=k_ind(t+1,1).*(k_ind(t+1,1)>=k_min).*(k_ind(t+1,1)<=k_max)+k_min*(k_ind(t+1,1)<k_min)+k_max*(k_ind(t+1,1)>k_max);
               % restrict k_ind to be in [k_min,k_max] range
               
    % Individual income
    income_ind(t,1)=k_ind(t,1)*interestrate(t)+(idshock(t,individual)-1).*l_bar*wage(t)+mu*(2-idshock(t,individual)).*wage(t)-...
    ur(agshock(t))/(1-ur(agshock(t)))*mu*(idshock(t,individual)-1).*wage(t)+redist*IHT*kmts(t)*p_death;

    % Individual consumption

    c_ind(t,1)=k_ind(t,1)*(1-delta+interestrate(t))+(idshock(t,individual)-1).*l_bar*wage(t)+mu*(2-idshock(t,individual)).*wage(t)-...
    ur(agshock(t))/(1-ur(agshock(t)))*mu*(idshock(t,individual)-1).*wage(t)-k_ind(t+1,1)+redist*IHT*kmts(t)*p_death;
               

    %else
       % k_ind(t+1,1)=(1-IHT)*k_ind(t,1);  
       % income_ind(t,1)=redist*IHT*kmts(t)*p_death;
       % c_ind(t,1)=redist*IHT*kmts(t)*p_death;
    %end
        
end 

range=[200 270];

figure(7)
subplot(6,1,1)
plot(a(agshock(range(1):range(2))))
title('Productivity')
set(gca,'FontUnits','points',...
'FontWeight','normal',...
'FontSize',11,...
'FontName','Times New Roman');
subplot(6,1,2)
plot(income_ind(range(1):range(2)))
title('Income')
set(gca,'FontUnits','points',...
'FontWeight','normal',...
'FontSize',11,...
'FontName','Times New Roman');
subplot(6,1,3)
plot(c_ind(range(1):range(2)))
title('Consumption')
set(gca,'FontUnits','points',...
'FontWeight','normal',...
'FontSize',11,...
'FontName','Times New Roman');
subplot(6,1,4)
plot(k_ind(range(1):range(2)))
title('Personal Savings')
set(gca,'FontUnits','points',...
'FontWeight','normal',...
'FontSize',11,...
'FontName','Times New Roman');
subplot(6,1,5)
plot(wage(range(1):range(2)))
title('Wage')
set(gca,'FontUnits','points',...
'FontWeight','normal',...
'FontSize',11,...
'FontName','Times New Roman');
yyaxis right
plot(interestrate(range(1):range(2)),'--')
title('Wage and Interest Rate')
set(gca,'FontUnits','points',...
'FontWeight','normal',...
'FontSize',11,...
'FontName','Times New Roman');
subplot(6,1,6)
plot(idshock(range(1):range(2),individual)-1);
title('Employment Status')
set(gca,'FontUnits','points',...
'FontWeight','normal',...
'FontSize',11,...
'FontName','Times New Roman');
print -dpdf -fillpage IndPanel
%% Policy Function
policy=ones(ngridk,ngridkm,nstates_ag,nstates_id,3);
i=1;
parfor i=1:ngridk
    for j=1:ngridkm
        for l=1:nstates_ag
            for m=1:nstates_id
                for p=1:3
                [check, policy(i,j,l,m,p)]=min(abs(k-kprime(i,j,l,m,p)));
                end
            end
        end
    end
end
%% Plotting Policy Functions

figure(18)
set(gcf,'Units','inches',...
'Position',[0 0 width height],...
'PaperPositionMode','auto','PaperOrientation','landscape','PaperUnits','Inches','PaperSize',[width, height]);
plot(k,kprime(:,3,2,2),'-',k,kprime(:,3,2,1),'--','LineWidth',2)
hold on
plot(k,k,'k','LineWidth',0.5)
hold off
xlabel('Current Wealth')
ylabel('Optimal Savings')
xlim([-11 40])
ylim([-11 40])
legend('Employed','Unemployed','45 degree line','location','northwest')
set(gca,'FontUnits','points',...
'FontWeight','normal',...
'FontSize',11,...
'FontName','Times New Roman');
print -dpdf -bestfit polk

figure(19)
set(gcf,'Units','inches',...
'Position',[0 0 width height],...
'PaperPositionMode','auto','PaperOrientation','landscape','PaperUnits','Inches','PaperSize',[width, height]);
plot(km,kprime(64,:,1,2),'-',km,kprime(64,:,1,1),'-o',km,kprime(64,:,2,2),'--',km,kprime(64,:,2,1),'--o','LineWidth',2)
xlabel('Aggregate capital')
ylabel('Optimal Savings')
legend('Bad, Employed','Bad, Unemployed','Good, Employed','Good, Unemployed','location','east')
set(gca,'FontUnits','points',...
'FontWeight','normal',...
'FontSize',11,...
'FontName','Times New Roman');
print -dpdf -bestfit polkm
%% Value Function
c=cat(5,c1,c2,c3);
%Vp=ones(ngridk,ngridkm,nstates_ag,nstates_id,3); 
%V=ones(ngridk,ngridkm,nstates_ag,nstates_id,3);   
load('values','V') % begin with last converged solution
Vp=V;
dif_v=1;
tol=1e-5;
Vq=ones(100,6,2);
Vs=ones(1,2);
if gamma==1
    util=log(c);
else
    util=(c.^(1-gamma)-1)/(1-gamma);
end
prob_u=[prob_bu prob_gu prob_u1u prob_u2u prob_b1u prob_g1u];
prob_up=prob_u(1,:,:,:);
prob_up1=[prob_up(1,1,:,:) prob_up(1,6,:,:) prob_up(1,11,:,:) prob_up(1,16,:,:) prob_up(1,21,:,:) prob_up(1,26,:,:)];
prob_u=squeeze(prob_up1);
prob_e=[prob_be prob_ge prob_u1e prob_u2e prob_b1e prob_g1e];
prob_ep=prob_e(1,:,:,:);
prob_ep1=[prob_ep(1,1,:,:) prob_ep(1,6,:,:) prob_ep(1,11,:,:) prob_ep(1,16,:,:) prob_ep(1,21,:,:) prob_ep(1,26,:,:)];
prob_e=squeeze(prob_ep1);

[V]=VFI(V,Vp,dif_v,util,tol,betas,p_death,prob_e,prob_u,...
    nstates_ag,nstates_id,ngridk,ngridkm,k,km,a2,epsilon2,kmprime,kprime);

%{
while dif_v>tol    
    tic
    for p=1:3
        discount=betas(p)*(1-p_death);
        for l=1:nstates_ag
            for j=1:ngridkm
                Vq=reshape(interpn(k,km,a2,epsilon2,V(:,:,:,:,p),k,kmprime(1,j,l,1),a2,epsilon2,'spline'),[ngridk,nstates_ag,nstates_id]);
                for m=1:nstates_id
                    for i=1:ngridk
                        Vs=reshape(interpn(k,a2,epsilon2,Vq(:,:,:),kprime(i,j,l,m,p),a2,epsilon2,'spline'),[6,2]);            
                        Vp(i,j,l,m,p)=util(i,j,l,m,p)+discount*(prob_u(:,l,m)'*Vs(:,1)+prob_e(:,l,m)'*Vs(:,2));
                        %if Vp(i,j,l,m)<0
                        %    Vp(i,j,k,m)=0;
                        %end
                    end
                end
            end
        end
    end
    dif_v=max(max(max(max(max(abs(V-Vp)))))) %#ok<NOPTS>
    V=Vp;
    toc
end
%}
%{

while dif_v>tol    
    tic
    for i=1:ngridk
        for j=1:ngridkm
            for l=1:nstates_ag
                for m=1:nstates_id
                    Vs=reshape(interpn(k,km,a2,epsilon2,V(:,:,:,:),kprime(i,j,l,m),kmprime(i,j,l,m),a2,epsilon2,'spline'),[5,2]);            
                    Vp(i,j,l,m)=util(i,j,l,m)+discount*(prob_u(:,l,m)'*Vs(:,1)+prob_e(:,l,m)'*Vs(:,2));
                end
            end
        end
    end
    dif_v=max(max(max(max(abs(V-Vp)))))
    V=Vp;
    toc
end
Vp(i,j,l,m)=log(c(i,j,l,m))+...
                        (1-p_death)*beta*(...
                        prob_bu(i,j,l,m)*interpn(k,km,V(:,:,1,1),kprime(i,j,l,m),kmprime(i,j,l,m),'nearest') + ...
                        prob_be(i,j,l,m)*interpn(k,km,V(:,:,1,2),kprime(i,j,l,m),kmprime(i,j,l,m),'nearest')+...
                        prob_gu(i,j,l,m)*interpn(k,km,V(:,:,2,1),kprime(i,j,l,m),kmprime(i,j,l,m),'nearest')+...
                        prob_ge(i,j,l,m)*interpn(k,km,V(:,:,2,2),kprime(i,j,l,m),kmprime(i,j,l,m),'nearest')+...
                        prob_uu(i,j,l,m)*interpn(k,km,V(:,:,3,1),kprime(i,j,l,m),kmprime(i,j,l,m),'nearest')+...
                        prob_ue(i,j,l,m)*interpn(k,km,V(:,:,3,2),kprime(i,j,l,m),kmprime(i,j,l,m),'nearest')+...
                        prob_b1u(i,j,l,m)*interpn(k,km,V(:,:,4,1),kprime(i,j,l,m),kmprime(i,j,l,m),'nearest')+...
                        prob_b1e(i,j,l,m)*interpn(k,km,V(:,:,4,2),kprime(i,j,l,m),kmprime(i,j,l,m),'nearest')+...
                        prob_g1u(i,j,l,m)*interpn(k,km,V(:,:,5,1),kprime(i,j,l,m),kmprime(i,j,l,m),'nearest')+...
                        prob_g1e(i,j,l,m)*interpn(k,km,V(:,:,5,2),kprime(i,j,l,m),kmprime(i,j,l,m),'nearest'));

%}
save('dist','kcross1');
%% Testing Transitions

% testing transitions

T=10000;
N=10000;
[idshock,agshock,deathshock,betashock]  = SHOCKStransit(prob,T,N,ur_b,...
    betas,pi_beta,init_dcf,p_death,IHT);

[kmts,~,~]  = AGGREGATE_death(T,idshock,agshock,betashock,km_max,km_min,...
    kprime,km,k,epsilon2,k_min,k_max,kcross,a2,N,deathshock,redist,IHT,p_death,betas);

unemployment=2-mean(idshock,2);

figure(17)
set(gcf,'Units','inches',...
'Position',[0 0 width height],...
'PaperPositionMode','auto','PaperOrientation','landscape','PaperUnits','Inches','PaperSize',[width, height]);
Tts=1:1:T;
axis([1 2750 min(kmts)*0.99 max(kmts)*1.01])
plot (Tts,kmts(1:T,1),'-','LineWidth',2)
xlabel('Time')
ylabel('Aggregate Capital')
xlim([0 2750])
set(gca,'FontUnits','points',...
'FontWeight','normal',...
'FontSize',11,...
'FontName','Times New Roman');
print -dpdf -bestfit lomtrans
%% Bad History
% Parameters
T=1000;
N=10000;


% Generate the new simulated shocks
[idshock,agshock,deathshock,betashock]  = SHOCKStest(prob,T,N,ur_b,...
    betas,pi_beta,init_dcf,p_death,IHT);

% Create responses and aggregate variables
[kmts,kcross1,~]  = AGGREGATE_death(T,idshock,agshock,betashock,km_max,km_min,...
    kprime,km,k,epsilon2,k_min,k_max,kcross,a2,N,deathshock,redist,IHT,p_death,betas);

% Generating time series
kmalm=zeros(T,1);  
kmalm(1)=kmts(1);  
                                   
for t=1:T-1       % compute kmalm
   if agshock(t)==1
      kmalm(t+1)=exp(B(1)+B(2)*log(kmalm(t)));
   elseif agshock(t)==2
      kmalm(t+1)=exp(B(3)+B(4)*log(kmalm(t)));
   elseif agshock(t)==3
      kmalm(t+1)=exp(B(5)+B(6)*log(kmalm(t)));
   elseif agshock(t)==4
      kmalm(t+1)=exp(B(7)+B(8)*log(kmalm(t)));      
   elseif agshock(t)==5
      kmalm(t+1)=exp(B(9)+B(10)*log(kmalm(t)));
   else
      kmalm(t+1)=exp(B(11)+B(12)*log(kmalm(t)));
   end
end

% level of km at time of decision
capital=kmts(T);

% level of bribe
%allowbribe=0;
%bribe=0.0001;
%delta=allowbribe*bribe*ones(ngridk,1);

% Value functions in each possible state
V11=interpn(k,km,V(:,:,1,1),k,capital,'spline');
V12=interpn(k,km,V(:,:,1,2),k,capital,'spline');
V21=interpn(k,km,V(:,:,2,1),k,capital,'spline');
V22=interpn(k,km,V(:,:,2,2),k,capital,'spline');
V31=interpn(k,km,V(:,:,3,1),k,capital,'spline');
V32=interpn(k,km,V(:,:,3,2),k,capital,'spline');


% If Stay in current state
% If currently unemployed
V_unemployed2states=(prob(1,1)*V11+prob(1,2)*V12+prob(1,3)*V21+prob(1,4)*V22)/(sum(prob(1,1:4)));
V_unemployed1state=(prob(1,1)*V11+prob(1,2)*V12)/(sum(prob(1,1:2)));
V_unemployed=V11;
% If currently employed
V_employed2states=(prob(2,1)*V11+prob(2,2)*V12+prob(2,3)*V21+prob(2,4)*V22)/(sum(prob(2,1:4)));
V_employed1state=(prob(2,1)*V11+prob(2,2)*V12)/(sum(prob(2,1:2)));
V_employed=V12;

% If choose to move to uncertain bad state
% Expected value is probability move to the state, multiplied by the
% outcome in that state 
% E[v] for unemployed= prob move to u,u + prob move to u,e   

V_u=(prob(1,5)*V31+prob(1,6)*V32)/(prob(1,5)+prob(1,6));
V_e=(prob(2,5)*V31+prob(2,6)*V32)/(prob(2,5)+prob(2,6));


% Figure showing value functions in all 4 possible states
%{
figure(9)
plot(k,V_unemployed2states)
hold on
plot(k,V_employed2states)
plot(k,V_u)
plot(k,V_e)
legend('Initial unemployed','Initial employed','New for unemployed','New for employed','location', 'southeast')
xlabel('Individual Wealth')
ylabel('Value')
title('Value function comparison')
xlim([k_min,100])
%}

% Wealth distribution of employed and unemployed agents
j=1;
l=1;
for i=1:N
    if idshock(T,i)==2
        kcrosse(j,1)=kcross1(i); %#ok<SAGROW>
        j=j+1;
    else
        kcrossu(l,1)=kcross1(i); %#ok<SAGROW>
        l=l+1;
    end
end

% Plot of two overlayed weath distributions
%{
figure(10)
set(gcf,'Units','inches',...
'Position',[0 0 width height],...
'PaperPositionMode','auto','PaperOrientation','landscape','PaperUnits','Inches','PaperSize',[width, height]);
histogram(kcrosse,'BinWidth',5,'Normalization','probability')
hold on
histogram(kcrossu,'BinWidth',5,'Normalization','probability')
hold off
%title('Wealth Distributions')
xlabel('Wealth')
ylabel('Density')
legend('Employed Agents', 'Unemployed Agents')
xlim([-10 350])
ylim('auto')
set(gca,'FontUnits','points',...
'FontWeight','normal',...
'FontSize',11,...
'FontName','Times New Roman');
print -dpdf -bestfit testwealthdist
%}



% Expected benefit of moving to the uncertain state from 2 possibles
gaine1=V_e-V_employed2states;
gainu1=V_u-V_unemployed2states;

%{
figure(11) % Benfits if 2 possible states
yyaxis left
plot(k,gainu1)
ylabel('Value gain for unemployed')
hold on
yyaxis right
plot(k,gaine1)
plot(k,zeros(ngridk,1))
xlim([k_min max(kcross1)])
legend('gain for unemployed', 'gain for employed','location','southwest')
xlabel('Individual Wealth')
ylabel('Value gain for employed')
title('Value gain in next period for changing next period')
print('VoteLeave','-dpdf','-fillpage')
%}

% Expected benefit of moving to the uncertain state from 1 possible
gaine2=V_e-V_employed1state;
gainu2=V_u-V_unemployed1state;

figure(12) % Benfits if 1 possible state1
set(gcf,'Units','inches',...
'Position',[0 0 width height],...
'PaperPositionMode','auto','PaperOrientation','landscape','PaperUnits','Inches','PaperSize',[width, height]);
plot(k,gainu2,'-',k,gaine2,'--','LineWidth',2)
hold on
plot(k,zeros(ngridk,1),'k','LineWidth',0.5)
hold off
xlim([-10 max(kcross1)])
legend('Unemployed', 'Employed','location','east')
xlabel('Individual Wealth')
ylabel('Value gain')
set(gca,'FontUnits','points',...
'FontWeight','normal',...
'FontSize',11,...
'FontName','Times New Roman');
print -dpdf -bestfit ValueGainBadHist


% Expected benefit of moving to the uncertain state
gaine3=V_e-V_employed;
gainu3=V_u-V_unemployed;

%{
figure(13) 
yyaxis left
plot(k,gainu3)
hold on
yyaxis right
plot(k,gaine3)
plot(k,zeros(ngridk,1))
xlim([k_min max(kcross1)])
legend('gain for unemployed', 'gain for employed','location','southwest')
xlabel('Individual Wealth')
ylabel('Value gain for employed')
title('Value gain from only changing aggregate state')
print('Vote Leave','-dpdf','-fillpage')
%}

change=zeros(1,N);

for i=1:N
    if idshock(T,i)==2 % the employed
        if interpn(k,gaine2,kcross1(i))>=0
            change(i)=1;
        else
            change(i)=0;
        end
    else  % unemployed
        if interpn(k,gainu2,kcross1(i))>=0
            change(i)=1;
        else
            change(i)=0;
        end
    end
end
SocialGainBad=0;
for i=1:N
    if idshock(T,i)==2 % the employed
        SocialGainBad=SocialGainBad+interpn(k,gaine2,kcross1(i));
    else  % unemployed
        SocialGainBad=SocialGainBad+interpn(k,gainu2,kcross1(i));
    end
end

votes_for_movebad=sum(change)/N;
disp votes_for_movebad
disp SocialGainBad


save('values','V','B');



low=find(gaine2>=0,1,'first');
high=find(gaine2>=0,1,'last');

lo = k(low);
hi = k(high);

[n, xout] = hist(kcrosse,100);
n=n/N;
n1 = n(xout>lo&xout<hi);
x1 = xout(xout>lo&xout<hi);
n2 = n(xout<lo|xout>hi);
x2 = xout(xout<lo|xout>hi);

figure(14)
set(gcf,'Units','inches',...
'Position',[0 0 width height],...
'PaperPositionMode','auto','PaperOrientation','landscape','PaperUnits','Inches','PaperSize',[width, height]);
bar(x1,n1,'b','BarWidth', 1)
hold on
bar(x2,n2,'r','BarWidth', 1)
hold off
xlim([-20 800])
xlabel('Wealth')
ylabel('Density')
legend('Vote to Leave','Vote to Remain')
set(gca,'FontUnits','points',...
'FontWeight','normal',...
'FontSize',11,...
'FontName','Times New Roman');
print -dpdf -bestfit VoteBadHist
%% For Tables1

sorted_wealth=sort(kcross1)';

percentages=[0.01 0.25 0.50 0.75 0.99]';
nperc=length(percentages);
wealthatperc=ones(nperc,1);
for i=1:nperc
    wealthatperc(i)=sorted_wealth(percentages(i).*N);
end
VFgainu1=ones(nperc,1);
for i=1:nperc
    VFgainu1(i)=interpn(k,gainu2,wealthatperc(i),'spline');
end
VFgaine1=ones(nperc,1);
for i=1:nperc
    VFgaine1(i)=interpn(k,gaine2,wealthatperc(i),'spline');
end

% Expected value function at percentages if move
Euatperc=ones(nperc,1);
for i=1:nperc
    Euatperc(i)=interpn(k,V_u,wealthatperc(i),'spline');
end
Eeatperc=ones(nperc,1);
for i=1:nperc
    Eeatperc(i)=interpn(k,V_e,wealthatperc(i),'spline');
end


neededwealthu=ones(nperc,1);
for i=1:nperc
    neededwealthu(i)=interp1(V_unemployed1state,k,Euatperc(i),'spline');
end

neededwealthe=ones(nperc,1);
for i=1:nperc
    neededwealthe(i)=interp1(V_employed1state,k,Eeatperc(i),'spline');
end

neededincinwealthu1=neededwealthu-wealthatperc;
neededincinwealthe1=neededwealthe-wealthatperc;
%% Good History

% Generate the new simulated shocks
[idshock,agshock,deathshock,betashock]  = SHOCKStest2(prob,T,N,ur_b,...
    betas,pi_beta,init_dcf,p_death,IHT);

% Create responses and aggregate variables
[kmts,kcross1,savings]  = AGGREGATE_death(T,idshock,agshock,betashock,km_max,km_min,...
    kprime,km,k,epsilon2,k_min,k_max,kcross,a2,N,deathshock,redist,IHT,p_death,betas);

% Generating time series
kmalm=zeros(T,1);  
kmalm(1)=kmts(1);  
                                   
for t=1:T-1       % compute kmalm
   if agshock(t)==1
      kmalm(t+1)=exp(B(1)+B(2)*log(kmalm(t)));
   elseif agshock(t)==2
      kmalm(t+1)=exp(B(3)+B(4)*log(kmalm(t)));
   elseif agshock(t)==3
      kmalm(t+1)=exp(B(5)+B(6)*log(kmalm(t)));
   elseif agshock(t)==4
      kmalm(t+1)=exp(B(7)+B(8)*log(kmalm(t)));      
   elseif agshock(t)==5
      kmalm(t+1)=exp(B(9)+B(10)*log(kmalm(t)));
   else
      kmalm(t+1)=exp(B(11)+B(12)*log(kmalm(t)));
   end
end

capital=kmts(T);
%capital=30;

% level of bribe
allowbribe=0;
bribe=0.0001;
delta=allowbribe*bribe*ones(ngridk,1);

% Value functions in each possible state
V11=interpn(k,km,V(:,:,1,1),k,capital,'spline');
V12=interpn(k,km,V(:,:,1,2),k,capital,'spline');
V21=interpn(k,km,V(:,:,2,1),k,capital,'spline');
V22=interpn(k,km,V(:,:,2,2),k,capital,'spline');
V31=interpn(k,km,V(:,:,3,1),k,capital,'spline');
V32=interpn(k,km,V(:,:,3,2),k,capital,'spline');
V41=interpn(k,km,V(:,:,4,1),k,capital,'spline');
V42=interpn(k,km,V(:,:,4,2),k,capital,'spline');

% If Stay in current state
% If currently unemployed
V_unemployed2states=(prob(1,1)*V11+prob(1,2)*V12+prob(1,3)*V21+prob(1,4)*V22)/(sum(prob(1,1:4)));
V_unemployed1state=(prob(3,3)*V21+prob(3,4)*V22)/(sum(prob(3,3:4)));
V_unemployed=V11;
% If currently employed
V_employed2states=(prob(2,1)*V11+prob(2,2)*V12+prob(2,3)*V21+prob(2,4)*V22)/(sum(prob(2,1:4)));
V_employed1state=(prob(4,3)*V21+prob(4,4)*V22)/(sum(prob(4,3:4)));
V_employed=V12;

% If choose to move to uncertain bad state
% Expected value is probability move to the state, multiplied by the
% outcome in that state 
% E[v] for unemployed= prob move to u,u + prob move to u,e   

V_u=(prob(3,7)*V41+prob(3,8)*V42)/(prob(3,7)+prob(3,8));
V_e=(prob(4,7)*V41+prob(4,8)*V42)/(prob(4,7)+prob(4,8));

% Expected benefit of moving to the uncertain state from 1 possible
gaine2=V_e-V_employed1state;
gainu2=V_u-V_unemployed1state;

figure(22) % Benfits if 1 possible state1
set(gcf,'Units','inches',...
'Position',[0 0 width height],...
'PaperPositionMode','auto','PaperOrientation','landscape','PaperUnits','Inches','PaperSize',[width, height]);
plot(k,gainu2,'-',k,gaine2,'--','LineWidth',2)
hold on
plot(k,zeros(ngridk,1),'k','LineWidth',0.5)
hold off
xlim([-9 max(kcross1)])
legend('Gain for Unemployed', 'Gain for Employed','location','east')
xlabel('Individual Wealth')
ylabel('Value gain for employed')
set(gca,'FontUnits','points',...
'FontWeight','normal',...
'FontSize',11,...
'FontName','Times New Roman');
print -dpdf -bestfit ValueGainGoodHist
%% Tables2

sorted_wealth=sort(kcross1)';

percentages=[0.01 0.25 0.50 0.75 0.99]';
nperc=length(percentages);
wealthatperc=ones(nperc,1);
for i=1:nperc
    wealthatperc(i)=sorted_wealth(percentages(i).*N);
end
VFgainu2=ones(nperc,1);
for i=1:nperc
    VFgainu2(i)=interpn(k,gainu2,wealthatperc(i),'spline');
end
VFgaine2=ones(nperc,1);
for i=1:nperc
    VFgaine2(i)=interpn(k,gaine2,wealthatperc(i),'spline');
end

% Expected value function at percentages if move
Euatperc=ones(nperc,1);
for i=1:nperc
    Euatperc(i)=interpn(k,V_u,wealthatperc(i),'spline');
end
Eeatperc=ones(nperc,1);
for i=1:nperc
    Eeatperc(i)=interpn(k,V_e,wealthatperc(i),'spline');
end

neededwealthu=ones(nperc,1);
for i=1:nperc
    neededwealthu(i)=interp1(V_unemployed1state,k,Euatperc(i),'spline');
end

neededwealthe=ones(nperc,1);
for i=1:nperc
    neededwealthe(i)=interp1(V_employed1state,k,Eeatperc(i),'spline');
end

neededincinwealthu2=neededwealthu-wealthatperc;
neededincinwealthe2=neededwealthe-wealthatperc;

for i=1:N
    if idshock(T,i)==2
        kcrosse(j,1)=kcross1(i); 
        j=j+1;
    else
        kcrossu(l,1)=kcross1(i); 
        l=l+1;
    end
end

changegood=zeros(1,N);

for i=1:N
    if idshock(T,i)==2 % the employed
        if interpn(k,gaine2,kcross1(i))>=0
            change(i)=1;
        else
            change(i)=0;
        end
    else  % unemployed
        if interpn(k,gainu2,kcross1(i))>=0
            change(i)=1;
        else
            change(i)=0;
        end
    end
end

votes_for_movegood=sum(change)/N %#ok<NOPTS>


low=find(gaine2>=0,1,'first');
high=find(gaine2>=0,1,'last');

lo = k(low);
hi = k(high);

[n, xout] = hist(kcrosse,100);
n=n/N;
n1 = n(xout>lo&xout<hi);
x1 = xout(xout>lo&xout<hi);
n2 = n(xout<lo|xout>hi);
x2 = xout(xout<lo|xout>hi);

figure(14)
set(gcf,'Units','inches',...
'Position',[0 0 width height],...
'PaperPositionMode','auto','PaperOrientation','landscape','PaperUnits','Inches','PaperSize',[width, height]);
bar(x1,n1,'b','BarWidth', 1)
hold on
bar(x2,n2,'r','BarWidth', 1)
hold off
xlim([-20 800])
xlabel('Wealth')
ylabel('Density')
legend('Vote to Leave','Vote to Remain')
set(gca,'FontUnits','points',...
'FontWeight','normal',...
'FontSize',11,...
'FontName','Times New Roman');
print -dpdf -bestfit VoteGoodHist
%% Print Table Results

num2str(VFgainu1,'%10.3e')

T1 = table(1000.*VFgainu1,1000.*VFgaine1,1000.*VFgainu2,1000.*VFgaine2) %#ok<NOPTS>
T2 = table(100*neededincinwealthu1,100*neededincinwealthe1,100*neededincinwealthu2,100*neededincinwealthe2) %#ok<NOPTS>
%% Good History, bad state

% Generate the new simulated shocks
[idshock,agshock,deathshock,betashock]  = SHOCKStest3(prob,T,N,ur_b,...
    betas,pi_beta,init_dcf,p_death,IHT);

% Create responses and aggregate variables
[kmts,kcross1,savings]  = AGGREGATE_death(T,idshock,agshock,betashock,km_max,km_min,...
    kprime,km,k,epsilon2,k_min,k_max,kcross,a2,N,deathshock,redist,IHT,p_death,betas);

% Generating time series
kmalm=zeros(T,1); 
kmalm(1)=kmts(1);  
                                   
for t=1:T-1       % compute kmalm
   if agshock(t)==1
      kmalm(t+1)=exp(B(1)+B(2)*log(kmalm(t)));
   elseif agshock(t)==2
      kmalm(t+1)=exp(B(3)+B(4)*log(kmalm(t)));
   elseif agshock(t)==3
      kmalm(t+1)=exp(B(5)+B(6)*log(kmalm(t)));
   elseif agshock(t)==4
      kmalm(t+1)=exp(B(7)+B(8)*log(kmalm(t)));      
   elseif agshock(t)==5
      kmalm(t+1)=exp(B(9)+B(10)*log(kmalm(t)));
   else
      kmalm(t+1)=exp(B(11)+B(12)*log(kmalm(t)));
   end
end

capital=kmts(T);
%capital=30;

% level of bribe
allowbribe=0;
bribe=0.0001;
delta=allowbribe*bribe*ones(ngridk,1);

% Value functions in each possible state
V11=interpn(k,km,V(:,:,1,1),k,capital,'spline');
V12=interpn(k,km,V(:,:,1,2),k,capital,'spline');
V21=interpn(k,km,V(:,:,2,1),k,capital,'spline');
V22=interpn(k,km,V(:,:,2,2),k,capital,'spline');
V31=interpn(k,km,V(:,:,3,1),k,capital,'spline');
V32=interpn(k,km,V(:,:,3,2),k,capital,'spline');


% If Stay in current state
% If currently unemployed
V_unemployed2states=(prob(1,1)*V11+prob(1,2)*V12+prob(1,3)*V21+prob(1,4)*V22)/(sum(prob(1,1:4)));
V_unemployed1state=(prob(1,1)*V11+prob(1,2)*V12)/(sum(prob(1,1:2)));
V_unemployed=V11;
% If currently employed
V_employed2states=(prob(2,1)*V11+prob(2,2)*V12+prob(2,3)*V21+prob(2,4)*V22)/(sum(prob(2,1:4)));
V_employed1state=(prob(2,1)*V11+prob(2,2)*V12)/(sum(prob(2,1:2)));
V_employed=V12;

% If choose to move to uncertain bad state
% Expected value is probability move to the state, multiplied by the
% outcome in that state 
% E[v] for unemployed= prob move to u,u + prob move to u,e   

V_u=(prob(1,5)*V31+prob(1,6)*V32)/(prob(1,5)+prob(1,6));
V_e=(prob(2,5)*V31+prob(2,6)*V32)/(prob(2,5)+prob(2,6));


% Wealth distribution of employed and unemployed agents
j=1;
l=1;
for i=1:N
    if idshock(T,i)==2
        kcrosse(j,1)=kcross1(i); %#ok<SAGROW>
        j=j+1;
    else
        kcrossu(l,1)=kcross1(i); %#ok<SAGROW>
        l=l+1;
    end
end


% Expected benefit of moving to the uncertain state from 2 possibles
gaine1=V_e-V_employed2states;
gainu1=V_u-V_unemployed2states;


% Expected benefit of moving to the uncertain state from 1 possible
gaine2=V_e-V_employed1state;
gainu2=V_u-V_unemployed1state;

figure(16) % Benfits if 1 possible state1
set(gcf,'Units','inches',...
'Position',[0 0 width height],...
'PaperPositionMode','auto','PaperOrientation','landscape','PaperUnits','Inches','PaperSize',[width, height]);
plot(k,gainu2,'-',k,gaine2,'--','LineWidth',2)
hold on
plot(k,zeros(ngridk,1),'k','LineWidth',0.5)
hold off
xlim([-10 max(kcross1)])
legend('Unemployed', 'Employed','location','east')
xlabel('Individual Wealth')
ylabel('Value gain')
set(gca,'FontUnits','points',...
'FontWeight','normal',...
'FontSize',11,...
'FontName','Times New Roman');
print -dpdf -bestfit ValueGainMixedHist


% Expected benefit of moving to the uncertain state
%gaine3=V_e-V_employed;
%gainu3=V_u-V_unemployed;

change=zeros(1,N);

for i=1:N
    if idshock(T,i)==2 % the employed
        if interpn(k,gaine2,kcross1(i))>=0
            change(i)=1;
        else
            change(i)=0;
        end
    else  % unemployed
        if interpn(k,gainu2,kcross1(i))>=0
            change(i)=1;
        else
            change(i)=0;
        end
    end
end
SocialGainBad=0;
for i=1:N
    if idshock(T,i)==2 % the employed
        SocialGainBad=SocialGainBad+interpn(k,gaine2,kcross1(i));
    else  % unemployed
        SocialGainBad=SocialGainBad+interpn(k,gainu2,kcross1(i));
    end
end

votes_for_movebad=sum(change)/N
disp votes_for_movebad
disp SocialGainBad

low=find(gaine2>=0,1,'first');
high=find(gaine2>=0,1,'last');

lo = k(low);
hi = k(high);

[n, xout] = hist(kcrosse,100);
n=n/N;
n1 = n(xout>lo&xout<hi);
x1 = xout(xout>lo&xout<hi);
n2 = n(xout<lo|xout>hi);
x2 = xout(xout<lo|xout>hi);

figure(17)
set(gcf,'Units','inches',...
'Position',[0 0 width height],...
'PaperPositionMode','auto','PaperOrientation','landscape','PaperUnits','Inches','PaperSize',[width, height]);
bar(x1,n1,'b','BarWidth', 1)
hold on
bar(x2,n2,'r','BarWidth', 1)
hold off
xlim([-20 800])
xlabel('Wealth')
ylabel('Density')
legend('Vote to Leave','Vote to Remain')
set(gca,'FontUnits','points',...
'FontWeight','normal',...
'FontSize',11,...
'FontName','Times New Roman');
print -dpdf -bestfit VoteMixedHist
%% Tables3
sorted_wealth=sort(kcross1)';

percentages=[0.01 0.25 0.50 0.75 0.99]';
nperc=length(percentages);
wealthatperc=ones(nperc,1);
for i=1:nperc
    wealthatperc(i)=sorted_wealth(percentages(i).*N);
end
VFgainu3=ones(nperc,1);
for i=1:nperc
    VFgainu3(i)=interpn(k,gainu2,wealthatperc(i),'spline');
end
VFgaine3=ones(nperc,1);
for i=1:nperc
    VFgaine3(i)=interpn(k,gaine2,wealthatperc(i),'spline');
end

% Expected value function at percentages if move
Euatperc=ones(nperc,1);
for i=1:nperc
    Euatperc(i)=interpn(k,V_u,wealthatperc(i),'spline');
end
Eeatperc=ones(nperc,1);
for i=1:nperc
    Eeatperc(i)=interpn(k,V_e,wealthatperc(i),'spline');
end

neededwealthu=ones(nperc,1);
for i=1:nperc
    neededwealthu(i)=interp1(V_unemployed1state,k,Euatperc(i),'spline');
end

neededwealthe=ones(nperc,1);
for i=1:nperc
    neededwealthe(i)=interp1(V_employed1state,k,Eeatperc(i),'spline');
end

neededincinwealthu3=neededwealthu-wealthatperc;
neededincinwealthe3=neededwealthe-wealthatperc;

T3=table(1000*VFgainu3,1000*VFgaine3,100*neededincinwealthu3,100*neededincinwealthe3)
%% Wealth distribution changing over the economic cycle
T=5000;

% Series of Good shocks
%
% Generate the new simulated shocks
[idshock,agshock,deathshock,betashock]  = SHOCKStestG(prob,T,N,ur_b,...
    betas,pi_beta,init_dcf,p_death,IHT);

% Create responses and aggregate variables
[~,kcrossG,~]  = AGGREGATE_death(T,idshock,agshock,betashock,km_max,km_min,...
    kprime,km,k,epsilon2,k_min,k_max,kcross,a2,N,deathshock,redist,IHT,p_death,betas);

weal=sort(kcrossG);
totalwealth=sum(weal);
lorenzG=ones(N,1);
for i=1:N
lorenzG(i)=100*sum(weal(1:i))/totalwealth; 
end
pop=linspace(0,100,N);


% Series of bad shocks
%
% Generate the new simulated shocks
[idshock,agshock,deathshock,betashock]  = SHOCKStestB(prob,T,N,ur_b,...
    betas,pi_beta,init_dcf,p_death,IHT);

% Create responses and aggregate variables
[~,kcrossB,~]  = AGGREGATE_death(T,idshock,agshock,betashock,km_max,km_min,...
    kprime,km,k,epsilon2,k_min,k_max,kcross,a2,N,deathshock,redist,IHT,p_death,betas);


weal=sort(kcrossB);
totalwealth=sum(weal);
lorenzB=ones(N,1);
for i=1:N
lorenzB(i)=100*sum(weal(1:i))/totalwealth; 
end

% Series of Uncertain shocks
%
% Generate the new simulated shocks
[idshock,agshock,deathshock,betashock]  = SHOCKStestU(prob,T,N,ur_b,...
    betas,pi_beta,init_dcf,p_death,IHT);

% Create responses and aggregate variables
[~,kcrossU,~]  = AGGREGATE_death(T,idshock,agshock,betashock,km_max,km_min,...
    kprime,km,k,epsilon2,k_min,k_max,kcross,a2,N,deathshock,redist,IHT,p_death,betas);

weal=sort(kcrossU);
totalwealth=sum(weal);
lorenzU=ones(N,1);
for i=1:N
lorenzU(i)=100*sum(weal(1:i))/totalwealth; 
end

figure(15)
set(gcf,'Units','inches',...
'Position',[0 0 width height],...
'PaperPositionMode','auto','PaperOrientation','landscape','PaperUnits','Inches','PaperSize',[width, height]);
histogram(kcrossG,'BinWidth',10)
hold on
histogram(kcrossB,'BinWidth',10)
hold on
histogram(kcrossU,'BinWidth',10)
hold off
xlim([-20 350])
legend('Good', 'Bad', 'Uncertain')
xlabel('Wealth')
ylabel('Number of Agents')
set(gca,'FontUnits','points',...
'FontWeight','normal',...
'FontSize',11,...
'FontName','Times New Roman');
print -dpdf -bestfit changingwealthdist


figure(16)
set(gcf,'Units','inches',...
'Position',[0 0 width height],...
'PaperPositionMode','auto','PaperOrientation','landscape','PaperUnits','Inches','PaperSize',[width, height]);
plot(pop,lorenzG,'-',pop,lorenzB,'--',pop,lorenzU,':','LineWidth',2)
hold on
plot(cumpop,cumwealth,'k','LineWidth',0.5)
hold off
legend('Good','Bad','Uncertain','Data','location','northwest')
xlabel('Cumulative population (%)')
ylabel('Cumulative Share of Wealth (%)')
xlim([0 100]);
ylim([-5 100]);
set(gca,'FontUnits','points',...
'FontWeight','normal',...
'FontSize',11,...
'FontName','Times New Roman');
print -dpdf -bestfit changingLorenz
%% For wealth dist stats table

% Generate the new simulated shocks
[idshock,agshock,deathshock,betashock]  = SHOCKStestwealth(prob,T,N,ur_b,...
    betas,pi_beta,init_dcf,p_death,IHT);

% Create responses and aggregate variables
[~,kcrosstest,~]  = AGGREGATE_death(T,idshock,agshock,betashock,km_max,km_min,...
    kprime,km,k,epsilon2,k_min,k_max,kcross,a2,N,deathshock,redist,IHT,p_death,betas);

weal=sort(kcrosstest);
totalwealth=sum(weal);
lorenz=ones(N,1);
for i=1:N
lorenz(i)=100*sum(weal(1:i))/totalwealth; 
end

lorenz1=lorenz;
cd C:\Users\Tim\Dropbox\MatlabCodes\sandbox\onebeta
load('Solution_to_model_new.mat', 'lorenz')
cd C:\Users\Tim\Dropbox\MatlabCodes\sandbox\6statesstochasticbeta

figure(22)
set(gcf,'Units','inches',...
'Position',[0 0 width height],...
'PaperPositionMode','auto','PaperOrientation','landscape','PaperUnits','Inches','PaperSize',[width, height]);
plot(pop,lorenz,'-',pop,lorenz1,'--','LineWidth',2)
hold on
plot(cumpop,cumwealth,'k','LineWidth',0.5)
hold off
legend('Baseline','Stochastic beta','Data','location','northwest')
%title('Lorenz Curves')
xlabel('Cumulative population (%)')
ylabel('Cumulative Share of Wealth (%)')
xlim([0 100]);
ylim([-5 100]);
set(gca,'FontUnits','points',...
'FontWeight','normal',...
'FontSize',11,...
'FontName','Times New Roman');
print -dpdf -bestfit lorenzforstochasticbeta
%% Save "Solution_to_model"

save Solution_to_model_new;

if send==1
    r21=num2str(R2bad(1));
    r22=num2str(R2good(1));
    r23=num2str(R2unc1(1));
    r24=num2str(R2bad1(1));
    r25=num2str(R2good1(1));
    r2s=[r21 ', ' r22 ', ' r23 ', ' r24 ', ' r25];
    coef1=num2str(B(1:2));
    coef2=num2str(B(3:4));
    coef3=num2str(B(5:6));
    coef4=num2str(B(9:10));
    coef5=num2str(B(11:12));
    
    mess=[msg '. Coefficients are, good: ' coef1 ' bad: ' coef2 ', uncertain1: ' coef3 ', very bad: ' coef4 ', very good: ' coef5 '. With R-squareds of ' r2s ];
    
    send_mail_message('tb13877@my.bristol.ac.uk','Matlab Files',mess ,{'C:\Users\Tim\Dropbox\MatlabCodes\sandbox\6statesstochasticbeta\erglorenz.pdf'...
        'C:\Users\Tim\Dropbox\MatlabCodes\sandbox\6statesstochasticbeta\ergwealthdist.pdf'...
        'C:\Users\Tim\Dropbox\MatlabCodes\sandbox\6statesstochasticbeta\lom.pdf',...
        'C:\Users\Tim\Dropbox\MatlabCodes\sandbox\6statesstochasticbeta\Panel.pdf',...
        'C:\Users\Tim\Dropbox\MatlabCodes\sandbox\6statesstochasticbeta\IndPanel.pdf'});
end