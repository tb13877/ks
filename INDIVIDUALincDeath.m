
function [kprime,kmprime,kprime1,kprime2,kprime3,c1,c2,c3,prob_b1e,prob_b1u,prob_be,prob_bu,prob_g1e,prob_g1u,prob_ge,prob_gu,prob_u1e,prob_u1u,prob_u2u,prob_u2e]...
    = INDIVIDUALincDeath(prob,ur_b,ur_g,ur_u1,ur_u2,ur_g1,ur_b1,ngridk,ngridkm,nstates_ag,...
    nstates_id,k,km,er_b,er_g,er_u1,er_u2,er_b1,er_g1,a,epsilon,l_bar,alpha,delta,gamma,betas,mu,km_max,...
    km_min,kprime1,kprime2,kprime3,B,criter_k,k_min,k_max,update_k,p_death,redist,IHT)

disp('Generating Responses');

prob_bu=zeros(ngridk,ngridkm,nstates_ag,nstates_id); % for a bad agg. state 
                           % and unemployed idios. state in the next period
prob_be=zeros(ngridk,ngridkm,nstates_ag,nstates_id); % for a bad agg. state 
                           % and employed idios. state in the next period
prob_gu=zeros(ngridk,ngridkm,nstates_ag,nstates_id); % for a good agg. state 
                           % and unemployed idios. state in the next period
prob_ge=zeros(ngridk,ngridkm,nstates_ag,nstates_id); % for a good agg. state 
                           % and employed idios. state in the next period
prob_u1u=zeros(ngridk,ngridkm,nstates_ag,nstates_id); % for a unc agg. state 
                           % and unemployed idios. state in the next period
prob_u1e=zeros(ngridk,ngridkm,nstates_ag,nstates_id); % for a unc agg. state 
                           % and employed idios. state in the next period
prob_u2u=zeros(ngridk,ngridkm,nstates_ag,nstates_id); % for a unc agg. state 
                           % and unemployed idios. state in the next period
prob_u2e=zeros(ngridk,ngridkm,nstates_ag,nstates_id); % for a unc agg. state 
                           % and employed idios. state in the next period
prob_b1u=zeros(ngridk,ngridkm,nstates_ag,nstates_id); % for a v.good agg. state 
                           % and unemployed idios. state in the next period
prob_b1e=zeros(ngridk,ngridkm,nstates_ag,nstates_id); % for a v.good agg. state 
                           % and employed idios. state in the next period 
prob_g1u=zeros(ngridk,ngridkm,nstates_ag,nstates_id); % for a v.bad agg. state 
                           % and unemployed idios. state in the next period
prob_g1e=zeros(ngridk,ngridkm,nstates_ag,nstates_id); % for a v.bad agg. state 
                           % and employed idios. state in the next period                           
%% Prob move to bu
prob_bu(:,:,1,1)=prob(1,1)*ones(ngridk,ngridkm);
prob_bu(:,:,1,2)=prob(2,1)*ones(ngridk,ngridkm);
prob_bu(:,:,2,1)=prob(3,1)*ones(ngridk,ngridkm);
prob_bu(:,:,2,2)=prob(4,1)*ones(ngridk,ngridkm);
prob_bu(:,:,3,1)=prob(5,1)*ones(ngridk,ngridkm);
prob_bu(:,:,3,2)=prob(6,1)*ones(ngridk,ngridkm);
prob_bu(:,:,4,1)=prob(7,1)*ones(ngridk,ngridkm);
prob_bu(:,:,4,2)=prob(8,1)*ones(ngridk,ngridkm);
prob_bu(:,:,5,1)=prob(9,1)*ones(ngridk,ngridkm);
prob_bu(:,:,5,2)=prob(10,1)*ones(ngridk,ngridkm);
prob_bu(:,:,6,1)=prob(11,1)*ones(ngridk,ngridkm);
prob_bu(:,:,6,2)=prob(12,1)*ones(ngridk,ngridkm);
%% Prob move to be
prob_be(:,:,1,1)=prob(1,2)*ones(ngridk,ngridkm);
prob_be(:,:,1,2)=prob(2,2)*ones(ngridk,ngridkm);
prob_be(:,:,2,1)=prob(3,2)*ones(ngridk,ngridkm);
prob_be(:,:,2,2)=prob(4,2)*ones(ngridk,ngridkm);
prob_be(:,:,3,1)=prob(5,2)*ones(ngridk,ngridkm);
prob_be(:,:,3,2)=prob(6,2)*ones(ngridk,ngridkm);
prob_be(:,:,4,1)=prob(7,2)*ones(ngridk,ngridkm);
prob_be(:,:,4,2)=prob(8,2)*ones(ngridk,ngridkm);
prob_be(:,:,5,1)=prob(9,2)*ones(ngridk,ngridkm);
prob_be(:,:,5,2)=prob(10,2)*ones(ngridk,ngridkm);
prob_bu(:,:,6,1)=prob(11,2)*ones(ngridk,ngridkm);
prob_bu(:,:,6,2)=prob(12,2)*ones(ngridk,ngridkm);
%% Prob move to gu
prob_gu(:,:,1,1)=prob(1,3)*ones(ngridk,ngridkm);
prob_gu(:,:,1,2)=prob(2,3)*ones(ngridk,ngridkm);
prob_gu(:,:,2,1)=prob(3,3)*ones(ngridk,ngridkm);
prob_gu(:,:,2,2)=prob(4,3)*ones(ngridk,ngridkm);
prob_gu(:,:,3,1)=prob(5,3)*ones(ngridk,ngridkm);
prob_gu(:,:,3,2)=prob(6,3)*ones(ngridk,ngridkm);
prob_gu(:,:,4,1)=prob(7,3)*ones(ngridk,ngridkm);
prob_gu(:,:,4,2)=prob(8,3)*ones(ngridk,ngridkm);
prob_gu(:,:,5,1)=prob(9,3)*ones(ngridk,ngridkm);
prob_gu(:,:,5,2)=prob(10,3)*ones(ngridk,ngridkm);
prob_gu(:,:,6,1)=prob(11,3)*ones(ngridk,ngridkm);
prob_gu(:,:,6,2)=prob(12,3)*ones(ngridk,ngridkm);
%% Prob move to ge
prob_ge(:,:,1,1)=prob(1,4)*ones(ngridk,ngridkm);
prob_ge(:,:,1,2)=prob(2,4)*ones(ngridk,ngridkm);
prob_ge(:,:,2,1)=prob(3,4)*ones(ngridk,ngridkm);
prob_ge(:,:,2,2)=prob(4,4)*ones(ngridk,ngridkm);
prob_ge(:,:,3,1)=prob(5,4)*ones(ngridk,ngridkm);
prob_ge(:,:,3,2)=prob(6,4)*ones(ngridk,ngridkm);
prob_ge(:,:,4,1)=prob(7,4)*ones(ngridk,ngridkm);
prob_ge(:,:,4,2)=prob(8,4)*ones(ngridk,ngridkm);
prob_ge(:,:,5,1)=prob(9,4)*ones(ngridk,ngridkm);
prob_ge(:,:,5,2)=prob(10,4)*ones(ngridk,ngridkm);
prob_ge(:,:,6,1)=prob(11,4)*ones(ngridk,ngridkm);
prob_ge(:,:,6,2)=prob(12,4)*ones(ngridk,ngridkm);
%% prob move to u1u
prob_u1u(:,:,1,1)=prob(1,5)*ones(ngridk,ngridkm);
prob_u1u(:,:,1,2)=prob(2,5)*ones(ngridk,ngridkm);
prob_u1u(:,:,2,1)=prob(3,5)*ones(ngridk,ngridkm);
prob_u1u(:,:,2,2)=prob(4,5)*ones(ngridk,ngridkm);
prob_u1u(:,:,3,1)=prob(5,5)*ones(ngridk,ngridkm);
prob_u1u(:,:,3,2)=prob(6,5)*ones(ngridk,ngridkm);
prob_u1u(:,:,4,1)=prob(7,5)*ones(ngridk,ngridkm);
prob_u1u(:,:,4,2)=prob(8,5)*ones(ngridk,ngridkm);
prob_u1u(:,:,5,1)=prob(9,5)*ones(ngridk,ngridkm);
prob_u1u(:,:,5,2)=prob(10,5)*ones(ngridk,ngridkm);
prob_u1u(:,:,6,1)=prob(11,5)*ones(ngridk,ngridkm);
prob_u1u(:,:,6,2)=prob(12,5)*ones(ngridk,ngridkm);
%% Prob move to u1e
prob_u1e(:,:,1,1)=prob(1,6)*ones(ngridk,ngridkm);
prob_u1e(:,:,1,2)=prob(2,6)*ones(ngridk,ngridkm);
prob_u1e(:,:,2,1)=prob(3,6)*ones(ngridk,ngridkm);
prob_u1e(:,:,2,2)=prob(4,6)*ones(ngridk,ngridkm);
prob_u1e(:,:,3,1)=prob(5,6)*ones(ngridk,ngridkm);
prob_u1e(:,:,3,2)=prob(6,6)*ones(ngridk,ngridkm);
prob_u1e(:,:,4,1)=prob(7,6)*ones(ngridk,ngridkm);
prob_u1e(:,:,4,2)=prob(8,6)*ones(ngridk,ngridkm);
prob_u1e(:,:,5,1)=prob(9,6)*ones(ngridk,ngridkm);
prob_u1e(:,:,5,2)=prob(10,6)*ones(ngridk,ngridkm);
prob_u1e(:,:,6,1)=prob(11,6)*ones(ngridk,ngridkm);
prob_u1e(:,:,6,2)=prob(12,6)*ones(ngridk,ngridkm);
%% prob move to u2u
prob_u2u(:,:,1,1)=prob(1,7)*ones(ngridk,ngridkm);
prob_u2u(:,:,1,2)=prob(2,7)*ones(ngridk,ngridkm);
prob_u2u(:,:,2,1)=prob(3,7)*ones(ngridk,ngridkm);
prob_u2u(:,:,2,2)=prob(4,7)*ones(ngridk,ngridkm);
prob_u2u(:,:,3,1)=prob(5,7)*ones(ngridk,ngridkm);
prob_u2u(:,:,3,2)=prob(6,7)*ones(ngridk,ngridkm);
prob_u2u(:,:,4,1)=prob(7,7)*ones(ngridk,ngridkm);
prob_u2u(:,:,4,2)=prob(8,7)*ones(ngridk,ngridkm);
prob_u2u(:,:,5,1)=prob(9,7)*ones(ngridk,ngridkm);
prob_u2u(:,:,5,2)=prob(10,7)*ones(ngridk,ngridkm);
prob_u2u(:,:,6,1)=prob(11,7)*ones(ngridk,ngridkm);
prob_u2u(:,:,6,2)=prob(12,7)*ones(ngridk,ngridkm);
%% Prob move to u1e
prob_u2e(:,:,1,1)=prob(1,8)*ones(ngridk,ngridkm);
prob_u2e(:,:,1,2)=prob(2,8)*ones(ngridk,ngridkm);
prob_u2e(:,:,2,1)=prob(3,8)*ones(ngridk,ngridkm);
prob_u2e(:,:,2,2)=prob(4,8)*ones(ngridk,ngridkm);
prob_u2e(:,:,3,1)=prob(5,8)*ones(ngridk,ngridkm);
prob_u2e(:,:,3,2)=prob(6,8)*ones(ngridk,ngridkm);
prob_u2e(:,:,4,1)=prob(7,8)*ones(ngridk,ngridkm);
prob_u2e(:,:,4,2)=prob(8,8)*ones(ngridk,ngridkm);
prob_u2e(:,:,5,1)=prob(9,8)*ones(ngridk,ngridkm);
prob_u2e(:,:,5,2)=prob(10,8)*ones(ngridk,ngridkm);
prob_u2e(:,:,6,1)=prob(11,8)*ones(ngridk,ngridkm);
prob_u2e(:,:,6,2)=prob(12,8)*ones(ngridk,ngridkm);
%% Prob move to b1u
prob_b1u(:,:,1,1)=prob(1,9)*ones(ngridk,ngridkm);
prob_b1u(:,:,1,2)=prob(2,9)*ones(ngridk,ngridkm);
prob_b1u(:,:,2,1)=prob(3,9)*ones(ngridk,ngridkm);
prob_b1u(:,:,2,2)=prob(4,9)*ones(ngridk,ngridkm);
prob_b1u(:,:,3,1)=prob(5,9)*ones(ngridk,ngridkm);
prob_b1u(:,:,3,2)=prob(6,9)*ones(ngridk,ngridkm);
prob_b1u(:,:,4,1)=prob(7,9)*ones(ngridk,ngridkm);
prob_b1u(:,:,4,2)=prob(8,9)*ones(ngridk,ngridkm);
prob_b1u(:,:,5,1)=prob(9,9)*ones(ngridk,ngridkm);
prob_b1u(:,:,5,2)=prob(10,9)*ones(ngridk,ngridkm);
prob_b1u(:,:,6,1)=prob(11,9)*ones(ngridk,ngridkm);
prob_b1u(:,:,6,2)=prob(12,9)*ones(ngridk,ngridkm);
%% Prob move to b1e
prob_b1e(:,:,1,1)=prob(1,10)*ones(ngridk,ngridkm);
prob_b1e(:,:,1,2)=prob(2,10)*ones(ngridk,ngridkm);
prob_b1e(:,:,2,1)=prob(3,10)*ones(ngridk,ngridkm);
prob_b1e(:,:,2,2)=prob(4,10)*ones(ngridk,ngridkm);
prob_b1e(:,:,3,1)=prob(5,10)*ones(ngridk,ngridkm);
prob_b1e(:,:,3,2)=prob(6,10)*ones(ngridk,ngridkm);
prob_b1e(:,:,4,1)=prob(7,10)*ones(ngridk,ngridkm);
prob_b1e(:,:,4,2)=prob(8,10)*ones(ngridk,ngridkm);
prob_b1e(:,:,5,1)=prob(9,10)*ones(ngridk,ngridkm);
prob_b1e(:,:,5,2)=prob(10,10)*ones(ngridk,ngridkm);
prob_b1e(:,:,6,1)=prob(11,10)*ones(ngridk,ngridkm);
prob_b1e(:,:,6,2)=prob(12,10)*ones(ngridk,ngridkm);
%% Prob move to g1u
prob_g1u(:,:,1,1)=prob(1,11)*ones(ngridk,ngridkm);
prob_g1u(:,:,1,2)=prob(2,11)*ones(ngridk,ngridkm);
prob_g1u(:,:,2,1)=prob(3,11)*ones(ngridk,ngridkm);
prob_g1u(:,:,2,2)=prob(4,11)*ones(ngridk,ngridkm);
prob_g1u(:,:,3,1)=prob(5,11)*ones(ngridk,ngridkm);
prob_g1u(:,:,3,2)=prob(6,11)*ones(ngridk,ngridkm);
prob_g1u(:,:,4,1)=prob(7,11)*ones(ngridk,ngridkm);
prob_g1u(:,:,4,2)=prob(8,11)*ones(ngridk,ngridkm);
prob_g1u(:,:,5,1)=prob(9,11)*ones(ngridk,ngridkm);
prob_g1u(:,:,5,2)=prob(10,11)*ones(ngridk,ngridkm);
prob_g1u(:,:,6,1)=prob(11,11)*ones(ngridk,ngridkm);
prob_g1u(:,:,6,2)=prob(12,11)*ones(ngridk,ngridkm);
%% prob move to g1e
prob_g1e(:,:,1,1)=prob(1,12)*ones(ngridk,ngridkm);
prob_g1e(:,:,1,2)=prob(2,12)*ones(ngridk,ngridkm);
prob_g1e(:,:,2,1)=prob(3,12)*ones(ngridk,ngridkm);
prob_g1e(:,:,2,2)=prob(4,12)*ones(ngridk,ngridkm);
prob_g1e(:,:,3,1)=prob(5,12)*ones(ngridk,ngridkm);
prob_g1e(:,:,3,2)=prob(6,12)*ones(ngridk,ngridkm);
prob_g1e(:,:,4,1)=prob(7,12)*ones(ngridk,ngridkm);
prob_g1e(:,:,4,2)=prob(8,12)*ones(ngridk,ngridkm);
prob_g1e(:,:,5,1)=prob(9,12)*ones(ngridk,ngridkm);
prob_g1e(:,:,5,2)=prob(10,12)*ones(ngridk,ngridkm);
prob_g1e(:,:,6,1)=prob(11,12)*ones(ngridk,ngridkm);
prob_g1e(:,:,6,2)=prob(12,12)*ones(ngridk,ngridkm);
%% Auxilary matrices (needed for computing interest rate, wage and wealth)  

kaux=zeros(ngridk,ngridkm,nstates_ag,nstates_id); % for individual capital
kaux(:,:,1,1)=k*ones(1,ngridkm);
kaux(:,:,1,2)=k*ones(1,ngridkm);
kaux(:,:,2,1)=k*ones(1,ngridkm);
kaux(:,:,2,2)=k*ones(1,ngridkm);
kaux(:,:,3,1)=k*ones(1,ngridkm);
kaux(:,:,3,2)=k*ones(1,ngridkm);
kaux(:,:,4,1)=k*ones(1,ngridkm);
kaux(:,:,4,2)=k*ones(1,ngridkm);
kaux(:,:,5,1)=k*ones(1,ngridkm);
kaux(:,:,5,2)=k*ones(1,ngridkm);
kaux(:,:,6,1)=k*ones(1,ngridkm);
kaux(:,:,6,2)=k*ones(1,ngridkm);

kmaux=zeros(ngridk,ngridkm,nstates_ag,nstates_id); % for the mean of capital 
                                                   % distribution (km)
kmaux(:,:,1,1)=ones(ngridk,1)*km';
kmaux(:,:,1,2)=ones(ngridk,1)*km';
kmaux(:,:,2,1)=ones(ngridk,1)*km';
kmaux(:,:,2,2)=ones(ngridk,1)*km';
kmaux(:,:,3,1)=ones(ngridk,1)*km';
kmaux(:,:,3,2)=ones(ngridk,1)*km';
kmaux(:,:,4,1)=ones(ngridk,1)*km';
kmaux(:,:,4,2)=ones(ngridk,1)*km';
kmaux(:,:,5,1)=ones(ngridk,1)*km';
kmaux(:,:,5,2)=ones(ngridk,1)*km';
kmaux(:,:,6,1)=ones(ngridk,1)*km';
kmaux(:,:,6,2)=ones(ngridk,1)*km';

aglabor=zeros(ngridk,ngridkm,nstates_ag,nstates_id); % for aggregate labor
aglabor(:,:,1,:)=er_b*ones(ngridk,ngridkm,nstates_id);
aglabor(:,:,2,:)=er_g*ones(ngridk,ngridkm,nstates_id);
aglabor(:,:,3,:)=er_u1*ones(ngridk,ngridkm,nstates_id);
aglabor(:,:,4,:)=er_u2*ones(ngridk,ngridkm,nstates_id);
aglabor(:,:,5,:)=er_b1*ones(ngridk,ngridkm,nstates_id);
aglabor(:,:,6,:)=er_g1*ones(ngridk,ngridkm,nstates_id);

agshock_aux=zeros(ngridk,ngridkm,nstates_ag,nstates_id); % for the aggregate 
                                                         % shock 
agshock_aux(:,:,1,:)=a(1)*ones(ngridk,ngridkm,nstates_id);
agshock_aux(:,:,2,:)=a(2)*ones(ngridk,ngridkm,nstates_id);
agshock_aux(:,:,3,:)=a(3)*ones(ngridk,ngridkm,nstates_id);
agshock_aux(:,:,4,:)=a(4)*ones(ngridk,ngridkm,nstates_id);
agshock_aux(:,:,5,:)=a(5)*ones(ngridk,ngridkm,nstates_id);
agshock_aux(:,:,6,:)=a(6)*ones(ngridk,ngridkm,nstates_id);

idshock_aux=zeros(ngridk,ngridkm,nstates_ag,nstates_id); % for the idiosyncratic 
                                                         % shock 
idshock_aux(:,:,:,1)=epsilon(1)*ones(ngridk,ngridkm,nstates_ag);
idshock_aux(:,:,:,2)=epsilon(2)*ones(ngridk,ngridkm,nstates_ag);
%% Interest rate, wage and wealth under given k and km

ones4=ones(ngridk,ngridkm,nstates_ag,nstates_id); 
                                                  
irateaux=alpha*(agshock_aux.*(kmaux./aglabor/l_bar).^(alpha-1));
wageaux=(1-alpha)*(agshock_aux.*(kmaux./aglabor/l_bar).^alpha);
                                         
redistaux=redist*IHT*kmaux*p_death;                                     
                                         
wealth=irateaux.*kaux+(wageaux.*idshock_aux)*l_bar+mu*(wageaux.*(ones4-idshock_aux))...
    +(1-delta)*kaux-mu*(wageaux.*(1-aglabor)./aglabor).*idshock_aux+redistaux; 

kmprime=zeros(ngridk,ngridkm,nstates_ag,nstates_id);

kmprime(:,:,1,1)=exp(B(1)*ones(ngridk,ngridkm)+B(2)*log(kmaux(:,:,1,1)));
kmprime(:,:,1,2)=exp(B(1)*ones(ngridk,ngridkm)+B(2)*log(kmaux(:,:,1,2)));
kmprime(:,:,2,1)=exp(B(3)*ones(ngridk,ngridkm)+B(4)*log(kmaux(:,:,2,1)));
kmprime(:,:,2,2)=exp(B(3)*ones(ngridk,ngridkm)+B(4)*log(kmaux(:,:,2,2)));
kmprime(:,:,3,1)=exp(B(5)*ones(ngridk,ngridkm)+B(6)*log(kmaux(:,:,3,1)));
kmprime(:,:,3,2)=exp(B(5)*ones(ngridk,ngridkm)+B(6)*log(kmaux(:,:,3,2)));
kmprime(:,:,4,1)=exp(B(7)*ones(ngridk,ngridkm)+B(8)*log(kmaux(:,:,4,1)));
kmprime(:,:,4,2)=exp(B(7)*ones(ngridk,ngridkm)+B(8)*log(kmaux(:,:,4,2)));
kmprime(:,:,5,1)=exp(B(9)*ones(ngridk,ngridkm)+B(10)*log(kmaux(:,:,5,1)));
kmprime(:,:,5,2)=exp(B(9)*ones(ngridk,ngridkm)+B(10)*log(kmaux(:,:,5,2)));
kmprime(:,:,6,1)=exp(B(11)*ones(ngridk,ngridkm)+B(12)*log(kmaux(:,:,6,1)));
kmprime(:,:,6,2)=exp(B(11)*ones(ngridk,ngridkm)+B(12)*log(kmaux(:,:,6,2)));
kmprime=(kmprime>=km_min).*(kmprime<=km_max).*kmprime+(kmprime<km_min)*km_min...
    +(kmprime>km_max)*km_max; % restricting km' to be in [km_min,km_max] range


irate_b=alpha*a(1)*((kmprime./(er_b*ones4*l_bar)).^(alpha-1));  
                                      % under a bad future aggregate state
irate_g=alpha*a(2)*((kmprime./(er_g*ones4*l_bar)).^(alpha-1));  
                                      % under a good future aggregate state
irate_u1=alpha*a(3)*((kmprime./(er_u1*ones4*l_bar)).^(alpha-1));  
                                      % under a unc future aggregate state
irate_u2=alpha*a(4)*((kmprime./(er_u2*ones4*l_bar)).^(alpha-1));  
                                      % under a v.bad future aggregate state                                      
irate_b1=alpha*a(5)*((kmprime./(er_b1*ones4*l_bar)).^(alpha-1));
irate_g1=alpha*a(6)*((kmprime./(er_g1*ones4*l_bar)).^(alpha-1));
                              

wage_b=(1-alpha)*a(1)*((kmprime./(er_b*ones4*l_bar)).^(alpha)); 
                                      % under a bad future aggregate state
wage_g=(1-alpha)*a(2)*((kmprime./(er_g*ones4*l_bar)).^(alpha)); 
                                      % under a good future aggregate state
wage_u1=(1-alpha)*a(3)*((kmprime./(er_u1*ones4*l_bar)).^(alpha)); 

wage_u2=(1-alpha)*a(4)*((kmprime./(er_u2*ones4*l_bar)).^(alpha)); 
                                      % under a bad future aggregate state
wage_b1=(1-alpha)*a(5)*((kmprime./(er_b1*ones4*l_bar)).^(alpha)); 
                                      % under a good future aggregate state
wage_g1=(1-alpha)*a(6)*((kmprime./(er_g1*ones4*l_bar)).^(alpha));                                      
%% SOLVING THE INDIVIDUAL PROBLEM
small=10^-10;
%dif_k1=10000;


for type=1:3
    if type==1
       kprime=kprime1;
    elseif type==2
        kprime=kprime2;
    else
        kprime=kprime3;
    end
dif_k=1;
       while dif_k>criter_k
   
   % 1 Bad aggregate state and unemployed idiosyncratic state 
   
     k2prime_bu=interpn(k,km,kprime(:,:,1,1),kprime,kmprime,'spline'); % Interpolating
     cprime_bu=irate_b.*kprime+mu*(wage_b.*ones4)+(1-delta)*kprime-k2prime_bu+redistaux;  % consumption                                                
     cprime_bu=(cprime_bu>0).*cprime_bu+(cprime_bu<=0)*small;  % constraining consumption to be positive
     muprime_bu=cprime_bu.^(-gamma); % marginal utility of consumption
   
   % 2 Bad aggregate state and employed idiosyncratic state
   
     k2prime_be=interpn(k,km,kprime(:,:,1,2),kprime,kmprime,'spline');
     cprime_be=irate_b.*kprime+wage_b.*(epsilon(2)*l_bar*ones4)+(1-delta)*kprime ...
      - mu*(wage_b.*((ur_b./(1-ur_b))*ones4))-k2prime_be+redistaux;
     cprime_be=(cprime_be>0).*cprime_be+(cprime_be<=0)*small;
     muprime_be=cprime_be.^(-gamma);
 
   % 3 Good aggregate state and unemployed idiosyncratic state
   
     k2prime_gu=interpn(k,km,kprime(:,:,2,1),kprime,kmprime,'spline');
     cprime_gu=irate_g.*kprime+mu*(wage_g.*ones4)+(1-delta)*kprime-k2prime_gu+redistaux;
     cprime_gu=(cprime_gu>0).*cprime_gu+(cprime_gu<=0)*small;
     muprime_gu=cprime_gu.^(-gamma);
   
   % 4 Good aggregate state and employed idiosyncratic state
   
     k2prime_ge=interpn(k,km,kprime(:,:,2,2),kprime,kmprime,'spline');
     cprime_ge=irate_g.*kprime+wage_g.*(epsilon(2)*l_bar*ones4)+(1-delta)*kprime...
         -mu*(wage_g.*((ur_g./(1-ur_g))*ones4))-k2prime_ge+redistaux;
     cprime_ge=(cprime_ge>0).*cprime_ge+(cprime_ge<=0)*small;
     muprime_ge=cprime_ge.^(-gamma);
     
   % 5 Unc1 aggregate state and unemployed idiosyncratic state
   
     k2prime_u1u=interpn(k,km,kprime(:,:,3,1),kprime,kmprime,'spline');
     cprime_u1u=irate_u1.*kprime+mu*(wage_u1.*ones4)+(1-delta)*kprime-k2prime_u1u+redistaux;
     cprime_u1u=(cprime_u1u>0).*cprime_u1u+(cprime_u1u<=0)*small;
     muprime_u1u=cprime_u1u.^(-gamma);
   
   % 6 Unc1 aggregate state and employed idiosyncratic state
   
     k2prime_u1e=interpn(k,km,kprime(:,:,3,2),kprime,kmprime,'spline');
     cprime_u1e=irate_u1.*kprime+wage_u1.*(epsilon(2)*l_bar*ones4)+(1-delta)*kprime...
         -mu*(wage_u1.*((ur_u1./(1-ur_u1))*ones4))-k2prime_u1e+redistaux;
     cprime_u1e=(cprime_u1e>0).*cprime_u1e+(cprime_u1e<=0)*small;
     muprime_u1e=cprime_u1e.^(-gamma);   
     
   % 5a Unc2 aggregate state and unemployed idiosyncratic state
   
     k2prime_u2u=interpn(k,km,kprime(:,:,4,1),kprime,kmprime,'spline');
     cprime_u2u=irate_u2.*kprime+mu*(wage_u2.*ones4)+(1-delta)*kprime-k2prime_u2u+redistaux;
     cprime_u2u=(cprime_u2u>0).*cprime_u2u+(cprime_u2u<=0)*small;
     muprime_u2u=cprime_u2u.^(-gamma);
   
   % 6a Unc2 aggregate state and employed idiosyncratic state
   
     k2prime_u2e=interpn(k,km,kprime(:,:,4,2),kprime,kmprime,'spline');
     cprime_u2e=irate_u2.*kprime+wage_u2.*(epsilon(2)*l_bar*ones4)+(1-delta)*kprime...
         -mu*(wage_u2.*((ur_u2./(1-ur_u2))*ones4))-k2prime_u2e+redistaux;
     cprime_u2e=(cprime_u2e>0).*cprime_u2e+(cprime_u2e<=0)*small;
     muprime_u2e=cprime_u2e.^(-gamma);     
     
   % 7 V.bad aggregate state and unemployed idiosyncratic state
   
     k2prime_b1u=interpn(k,km,kprime(:,:,5,1),kprime,kmprime,'spline');
     cprime_b1u=irate_b1.*kprime+mu*(wage_b1.*ones4)+(1-delta)*kprime-k2prime_b1u+redistaux;
     cprime_b1u=(cprime_b1u>0).*cprime_b1u+(cprime_b1u<=0)*small;
     muprime_b1u=cprime_b1u.^(-gamma);
   
   % 8 V.bad aggregate state and employed idiosyncratic state
   
     k2prime_b1e=interpn(k,km,kprime(:,:,5,2),kprime,kmprime,'spline');
     cprime_b1e=irate_b1.*kprime+wage_b1.*(epsilon(2)*l_bar*ones4)+(1-delta)*kprime...
         -mu*(wage_b1.*((ur_b1./(1-ur_b1))*ones4))-k2prime_b1e+redistaux;
     cprime_b1e=(cprime_b1e>0).*cprime_b1e+(cprime_b1e<=0)*small;
     muprime_b1e=cprime_b1e.^(-gamma);      
     
   % 9 V.Good aggregate state and unemployed idiosyncratic state
   
     k2prime_g1u=interpn(k,km,kprime(:,:,6,1),kprime,kmprime,'spline');
     cprime_g1u=irate_g1.*kprime+mu*(wage_g1.*ones4)+(1-delta)*kprime-k2prime_g1u+redistaux;
     cprime_g1u=(cprime_g1u>0).*cprime_g1u+(cprime_g1u<=0)*small;
     muprime_g1u=cprime_g1u.^(-gamma);
   
   % 10 V.Good aggregate state and employed idiosyncratic state
   
     k2prime_g1e=interpn(k,km,kprime(:,:,6,2),kprime,kmprime,'spline');
     cprime_g1e=irate_g1.*kprime+wage_g1.*(epsilon(2)*l_bar*ones4)+(1-delta)*kprime...
         -mu*(wage_g1.*((ur_g1./(1-ur_g1))*ones4))-k2prime_g1e+redistaux;
     cprime_g1e=(cprime_g1e>0).*cprime_g1e+(cprime_g1e<=0)*small;
     muprime_g1e=cprime_g1e.^(-gamma);     
         
   
   % Expectations
   
    expec=((muprime_bu.*((1-delta)*ones4+irate_b)).*prob_bu+...
    (muprime_be.*((1-delta)*ones4+irate_b)).*prob_be+...
    (muprime_gu.*((1-delta)*ones4+irate_g)).*prob_gu+...
    (muprime_ge.*((1-delta)*ones4+irate_g)).*prob_ge+...
    (muprime_u1u.*((1-delta)*ones4+irate_u1)).*prob_u1u+...
    (muprime_u1e.*((1-delta)*ones4+irate_u1)).*prob_u1e+...
    (muprime_u2u.*((1-delta)*ones4+irate_u2)).*prob_u2u+...
    (muprime_u2e.*((1-delta)*ones4+irate_u2)).*prob_u2e+...
    (muprime_b1u.*((1-delta)*ones4+irate_b1)).*prob_b1u+...
    (muprime_b1e.*((1-delta)*ones4+irate_b1)).*prob_b1e+...
    (muprime_g1u.*((1-delta)*ones4+irate_g1)).*prob_g1u+...
    (muprime_g1e.*((1-delta)*ones4+irate_g1)).*prob_g1e).*(1-p_death);
   
   cn=(betas(type)*expec).^(-1/gamma); 
                    
   kprimen=wealth-cn; % new policy rule
    
   kprimen=(kprimen>=k_min).*(kprimen<=k_max).*kprimen+(kprimen<k_min)*k_min...
       +(kprimen>k_max)*k_max; 
     
   dif_k=max(max(max(max(abs(kprimen-kprime))))); 
   
   kprime=update_k*kprimen+(1-update_k)*kprime; % updating policy rule     
        end
    if type==1
    kprime1=kprime;
    c1=cn;
    elseif type==2
    kprime2=kprime;
    c2=cn;
    else
    kprime3=kprime;
    c3=cn;
    end
end

% Consumption function

%c=wealth-kprime; % follows from the budget constraint
kprime=cat(5,kprime1,kprime2,kprime3);

