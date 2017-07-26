function [idshock,agshock,deathshock,betashock]  = SHOCKSincdeath(prob,T,N,ur_b,p_death,IHT,testtrans,period...
    ,betas,pi_beta,init_dcf)
disp('Generating shocks');

idshock=zeros(T,N); % matrix of idiosyncratic shocks 
agshock=zeros(T,1); % vector of aggregate shocks
deathshock=ones(T,N);
betashock=ones(T,N);

%% Transition probabilities between the aggregate states 

prob_ag=zeros(6,6);  
prob_ag(1,1)=prob(1,1)+prob(1,2); prob_ag(1,2)=prob(1,3)+prob(1,4); prob_ag(1,3)=prob(1,5)+prob(1,6);
prob_ag(1,4)=prob(1,7)+prob(1,8); prob_ag(1,5)=prob(1,9)+prob(1,10); prob_ag(1,6)=prob(1,11)+prob(1,12);

prob_ag(2,1)=prob(3,1)+prob(3,2); prob_ag(2,2)=prob(3,3)+prob(3,4); prob_ag(2,3)=prob(3,5)+prob(3,6);
prob_ag(2,4)=prob(3,7)+prob(3,8); prob_ag(2,5)=prob(3,9)+prob(3,10); prob_ag(2,6)=prob(3,11)+prob(3,12);

prob_ag(3,1)=prob(5,1)+prob(5,2); prob_ag(3,2)=prob(5,3)+prob(5,4); prob_ag(3,3)=prob(5,5)+prob(5,6);
prob_ag(3,4)=prob(5,7)+prob(5,8); prob_ag(3,5)=prob(5,9)+prob(5,10); prob_ag(3,6)=prob(5,11)+prob(5,12);

prob_ag(4,1)=prob(7,1)+prob(7,2); prob_ag(4,2)=prob(7,3)+prob(7,4); prob_ag(4,3)=prob(7,5)+prob(7,6);
prob_ag(4,4)=prob(7,7)+prob(7,8); prob_ag(4,5)=prob(7,9)+prob(7,10); prob_ag(4,6)=prob(7,11)+prob(7,12);

prob_ag(5,1)=prob(9,1)+prob(9,2); prob_ag(5,2)=prob(9,3)+prob(9,4); prob_ag(5,3)=prob(9,5)+prob(9,6);
prob_ag(5,4)=prob(9,7)+prob(9,8); prob_ag(5,5)=prob(9,9)+prob(9,10); prob_ag(5,6)=prob(9,11)+prob(9,12);

prob_ag(6,1)=prob(11,1)+prob(11,2); prob_ag(6,2)=prob(11,3)+prob(11,4); prob_ag(6,3)=prob(11,5)+prob(11,6);
prob_ag(6,4)=prob(11,7)+prob(11,8); prob_ag(6,5)=prob(11,9)+prob(11,10); prob_ag(6,6)=prob(11,11)+prob(11,12);
%% Probability of an idiosyncratic shock given  aggregate shock

p_bb_uu = prob(1,1)/prob_ag(1,1); %p_bb_ue=1-p_bb_uu;
p_bb_ee = prob(2,2)/prob_ag(1,1); %p_bb_eu=1-p_bb_ee;
p_bg_uu = prob(1,3)/prob_ag(1,2); %p_bg_ue=1-p_bg_uu;
p_bg_ee = prob(2,4)/prob_ag(1,2); %p_bg_eu=1-p_bg_ee;
p_bu1_uu = prob(1,5)/prob_ag(1,3); %p_bu_ue=1-p_bu_uu;
p_bu1_ee = prob(2,6)/prob_ag(1,3); %p_bg_eu=1-p_bu_ee;
p_bu2_uu = 0;
p_bu2_ee = 0;
p_bb1_uu = 0; %p_bb1_ue=0;
p_bb1_ee = 0; %p_bb1_eu=0;
p_bg1_uu = 0; %p_bg1_ue=0;
p_bg1_ee = 0; %p_bg1_eu=0;

p_gb_uu = prob(3,1)/prob_ag(2,1); %p_gb_ue=1-p_gb_uu;
p_gb_ee = prob(4,2)/prob_ag(2,1); %p_gb_eu=1-p_gb_ee;
p_gg_uu = prob(3,3)/prob_ag(2,2); %p_gg_ue=1-p_gg_uu;
p_gg_ee = prob(4,4)/prob_ag(2,2); %p_gg_eu=1-p_gg_ee;
p_gu1_uu = 0;
p_gu1_ee = 0;
p_gu2_uu = prob(3,7)/prob_ag(2,4); %p_gu_ue=1-p_gu_uu;
p_gu2_ee = prob(4,8)/prob_ag(2,4); %p_gu_eu=1-p_gu_ee;
p_gb1_uu = 0; %p_gb1_ue=0;
p_gb1_ee = 0; %p_gb1_eu=0;
p_gg1_uu = 0; %p_gg1_ue=0;
p_gg1_ee = 0; %p_gg1_eu=0;

p_u1b_uu = 0; %p_ub_ue = 0;
p_u1b_ee = 0; %p_ub_eu = 0;
p_u1g_uu = 0; %p_ub_ue = 0;
p_u1g_ee = 0; %p_ub_eu = 0;
p_u1u1_uu = prob(5,5)/prob_ag(3,3); %p_uu_ue=1-p_uu_uu;
p_u1u1_ee = prob(6,6)/prob_ag(3,3); %p_uu_eu=1-p_uu_ee;
p_u1u2_uu = 0;
p_u1u2_ee = 0;
p_u1b1_uu = prob(5,9)/prob_ag(3,5); %p_ub1_ue=1-p_ub1_uu;
p_u1b1_ee = prob(6,10)/prob_ag(3,5); %p_ub1_eu=1-p_ub1_ee;
p_u1g1_uu = prob(5,11)/prob_ag(3,6); %p_ug1_ue=1-p_ug1_uu;
p_u1g1_ee = prob(6,12)/prob_ag(3,6);%p_ug1_eu=1-p_ug1_ee;

p_u2b_uu = 0; %p_ub_ue = 0;
p_u2b_ee = 0; %p_ub_eu = 0;
p_u2g_uu = 0; %p_ub_ue = 0;
p_u2g_ee = 0; %p_ub_eu = 0;
p_u2u1_uu= 0;
p_u2u1_ee= 0;
p_u2u2_uu = prob(7,7)/prob_ag(4,4); %p_uu_ue=1-p_uu_uu;
p_u2u2_ee = prob(8,8)/prob_ag(4,4); %p_uu_eu=1-p_uu_ee;
p_u2b1_uu = prob(7,9)/prob_ag(4,5); %p_ub1_ue=1-p_ub1_uu;
p_u2b1_ee = prob(8,10)/prob_ag(4,5); %p_ub1_eu=1-p_ub1_ee;
p_u2g1_uu = prob(7,11)/prob_ag(4,6); %p_ug1_ue=1-p_ug1_uu;
p_u2g1_ee = prob(8,12)/prob_ag(4,6);%p_ug1_eu=1-p_ug1_ee;

p_b1b_uu = prob(9,1)/prob_ag(5,1); %p_b1b_ue=1-p_b1b_uu;
p_b1b_ee = prob(10,2)/prob_ag(5,1); %p_b1b_eu=1-p_b1b_ee;
p_b1g_uu = 0; %p_b1g_ue=0;
p_b1g_ee = 0; %p_b1g_eu=0;
p_b1u1_uu = 0; %p_b1u_ue=0;
p_b1u1_ee = 0; %p_b1u_eu=0;
p_b1u2_uu = 0;
p_b1u2_uu = 0;
p_b1b1_uu = prob(9,9)/prob_ag(5,5); %p_b1b1_ue=1-p_b1b1_uu;
p_b1b1_ee = prob(10,10)/prob_ag(5,5); %p_b1b1_eu=1-p_b1b1_ee;
p_b1g1_uu = prob(9,11)/prob_ag(5,6); %p_b1g1_ue=1-p_b1g1_uu;
p_b1g1_ee = prob(10,12)/prob_ag(5,6);% p_b1g1_eu=1-p_b1g1_ee;

p_g1b_uu = 0; %p_g1b_ue=0;
p_g1b_ee = 0; %p_g1b_eu=0;
p_g1g_uu = prob(11,3)/prob_ag(6,2); %p_g1g_ue=1-p_g1g_uu;
p_g1g_ee = prob(12,4)/prob_ag(6,2);% p_g1g_eu=1-p_g1g_ee;
p_g1u1_uu = 0; %p_g1u_ue=0;
p_g1u1_ee = 0; %p_g1u_eu=0;
p_g1u2_uu = 0;
p_g1u2_uu = 0;
p_g1b1_uu = prob(11,9)/prob_ag(6,5); %p_g1b1_ue=1-p_g1b1_uu;
p_g1b1_ee = prob(12,10)/prob_ag(6,5);% p_g1b1_eu=1-p_g1b1_ee;
p_g1g1_uu = prob(11,11)/prob_ag(6,6); %p_g1g1_ue=1-p_g1g1_uu;
p_g1g1_ee = prob(12,12)/prob_ag(6,6);% p_g1g1_eu=1-p_g1g1_ee;
%% Generation of the aggregate shocks 

agshock(1)=1; % start in a bad state 


for t=2:T
   raux=rand; 
   if raux<=prob_ag(agshock(t-1),1)  % probability of realising a bad shock in every initial state
      agshock(t)=1; % stay in bad state
   elseif raux<=prob_ag(agshock(t-1),1)+prob_ag(agshock(t-1),2)
      agshock(t)=2; % move to good state
   elseif raux<=prob_ag(agshock(t-1),1)+prob_ag(agshock(t-1),2)+prob_ag(agshock(t-1),3)
      agshock(t)=3; % move to unc state   
   elseif raux<=prob_ag(agshock(t-1),1)+prob_ag(agshock(t-1),2)+prob_ag(agshock(t-1),3)+prob_ag(agshock(t-1),4)
      agshock(t)=4; % move to v.bad state  
   elseif raux<=prob_ag(agshock(t-1),1)+prob_ag(agshock(t-1),2)+prob_ag(agshock(t-1),3)+prob_ag(agshock(t-1),4)+prob_ag(agshock(t-1),5)
      agshock(t)=5;
   else
      agshock(t)=6;
   end
end
if testtrans==1
    for t=1:T
        if t<period
            agshock(t)=1;
        elseif t<2*period
            agshock(t)=2;
        elseif t<3*period
            agshock(t)=3;
        elseif t<4*period
            agshock(t)=4;
        elseif t<5*period
            agshock(t)=5;
        end
    end
end
%% Generation of the idiosyncratic shocks for all agents in the first period

for i=1:N
   raux=rand;
   if raux<=ur_b 
      idshock(1,i)=1;
   else
      idshock(1,i)=2;
   end
end
%% Generation of the idiosyncratic shocks

for t=2:T
    
   for i=1:N
       raux=rand;
       if raux<p_death
           deathshock(t,i)=1-IHT;
       else
           deathshock(t,i)=1;
       end
   end
      
   if agshock(t-1)==1 && agshock(t)==1 % if the previous agg. shock was bad 
                                      % and the current agg. shock is bad
      for i=1:N
         raux=rand;
         if idshock(t-1,i)==1 % if the previous idiosyncratic shock was 1 
            if raux<=p_bb_uu
               idshock(t,i)=1;
            else
               idshock(t,i)=2;
            end
         else                 % if the previous idiosyncratic shock was 2 
            if raux<=p_bb_ee
               idshock(t,i)=2;
            else
               idshock(t,i)=1;
            end
         end
      end
   end
   
   if agshock(t-1)==1 && agshock(t)==2 % if the previous agg. shock was bad 
                                      % and the current agg. shock is good
      for i=1:N
         raux=rand;
         if idshock(t-1,i)==1 % remain in the bad idiosyncratic state
            if raux<=p_bg_uu
               idshock(t,i)=1;
            else
               idshock(t,i)=2;
            end
         else
            if raux<=p_bg_ee
               idshock(t,i)=2;
            else
               idshock(t,i)=1;
            end
         end
      end
   end
   
   if agshock(t-1)==2 && agshock(t)==1 % if the previous agg. shock was good 
                                      % and the current agg. shock is bad
      for i=1:N
         raux=rand;
         if idshock(t-1,i)==1
            if raux<=p_gb_uu
               idshock(t,i)=1;
            else
               idshock(t,i)=2;
            end
         else
            if raux<=p_gb_ee
               idshock(t,i)=2;
            else
               idshock(t,i)=1;
            end
         end
      end
   end
   
   if agshock(t-1)==2 && agshock(t)==2 % if the previous agg. shock was good 
                                      % and the current agg. shock is good
      for i=1:N
         raux=rand;
         if idshock(t-1,i)==1
            if raux<=p_gg_uu
               idshock(t,i)=1;
            else
               idshock(t,i)=2;
            end
         else
            if raux<=p_gg_ee
               idshock(t,i)=2;
            else
               idshock(t,i)=1;
            end
         end
      end
   end
   
   if agshock(t-1)==1 && agshock(t)==3 % if the previous agg. shock was bad 
                                      % and the current agg. shock is unc
      for i=1:N
         raux=rand;
         if idshock(t-1,i)==1 % if the previous idiosyncratic shock was 1 
            if raux<=p_bu1_uu
               idshock(t,i)=1;
            else
               idshock(t,i)=2;
            end
         else                 % if the previous idiosyncratic shock was 2 
            if raux<=p_bu1_ee
               idshock(t,i)=2;
            else
               idshock(t,i)=1;
            end
         end
      end
   end
   %{
   if agshock(t-1)==1 && agshock(t)==4 % if the previous agg. shock was bad 
                                      % and the current agg. shock is vbad
      for i=1:N
         raux=rand;
         if idshock(t-1,i)==1 % if the previous idiosyncratic shock was 1 
            if raux<=p_bb1_uu
               idshock(t,i)=1;
            else
               idshock(t,i)=2;
            end
         else                 % if the previous idiosyncratic shock was 2 
            if raux<=p_bb1_ee
               idshock(t,i)=2;
            else
               idshock(t,i)=1;
            end
         end
      end
   end
%}
   %{
   if agshock(t-1)==1 && agshock(t)==5 % if the previous agg. shock was bad 
                                      % and the current agg. shock is vgood
      for i=1:N
         raux=rand;
         if idshock(t-1,i)==1 % if the previous idiosyncratic shock was 1 
            if raux<=p_bg1_uu
               idshock(t,i)=1;
            else
               idshock(t,i)=2;
            end
         else                 % if the previous idiosyncratic shock was 2 
            if raux<=p_bg1_ee
               idshock(t,i)=2;
            else
               idshock(t,i)=1;
            end
         end
      end
   end
   %}
   if agshock(t-1)==2 && agshock(t)==4 % if the previous agg. shock was good 
                                      % and the current agg. shock is vbad
      for i=1:N
         raux=rand;
         if idshock(t-1,i)==1 % if the previous idiosyncratic shock was 1 
            if raux<=p_gu2_uu
               idshock(t,i)=1;
            else
               idshock(t,i)=2;
            end
         else                 % if the previous idiosyncratic shock was 2 
            if raux<=p_gu2_ee
               idshock(t,i)=2;
            else
               idshock(t,i)=1;
            end
         end
      end
   end
%{
   if agshock(t-1)==2 && agshock(t)==5 % if the previous agg. shock was good 
                                      % and the current agg. shock is vgood
      for i=1:N
         raux=rand;
         if idshock(t-1,i)==1 % if the previous idiosyncratic shock was 1 
            if raux<=p_gg1_uu
               idshock(t,i)=1;
            else
               idshock(t,i)=2;
            end
         else                 % if the previous idiosyncratic shock was 2 
            if raux<=p_gg1_ee
               idshock(t,i)=2;
            else
               idshock(t,i)=1;
            end
         end
      end
   end     
 %}  
   %{
   if agshock(t-1)==2 && agshock(t)==3 % if the previous agg. shock was good
                                      % and the current agg. shock is unc
      for i=1:N
         raux=rand;
         if idshock(t-1,i)==1 % if the previous idiosyncratic shock was 1 
            if raux<=p_gu_uu
               idshock(t,i)=1;
            else
               idshock(t,i)=2;
            end
         else                 % if the previous idiosyncratic shock was 2 
            if raux<=p_gu_ee
               idshock(t,i)=2;
            else
               idshock(t,i)=1;
            end
         end
      end
   end   
   %}
   %{
   if agshock(t-1)==3 && agshock(t)==1 % if the previous agg. shock was unc 
                                      % and the current agg. shock is bad
      for i=1:N
         raux=rand;
         if idshock(t-1,i)==1 % if the previous idiosyncratic shock was 1 
            if raux<=p_ub_uu
               idshock(t,i)=1;
            else
               idshock(t,i)=2;
            end
         else                 % if the previous idiosyncratic shock was 2 
            if raux<=p_ub_ee
               idshock(t,i)=2;
            else
               idshock(t,i)=1;
            end
         end
      end
   end
   %}
   %{
   if agshock(t-1)==3 && agshock(t)==2 % if the previous agg. shock was unc 
                                      % and the current agg. shock is good
      for i=1:N
         raux=rand;
         if idshock(t-1,i)==1 % remain in the bad idiosyncratic state
            if raux<=p_ug_uu
               idshock(t,i)=1;
            else
               idshock(t,i)=2;
            end
         else
            if raux<=p_ug_ee
               idshock(t,i)=2;
            else
               idshock(t,i)=1;
            end
         end
      end
   end   
   %}
    
   if agshock(t-1)==3 && agshock(t)==3 % if the previous agg. shock was unc 
                                      % and the current agg. shock is unc
      for i=1:N
         raux=rand;
         if idshock(t-1,i)==1 % if the previous idiosyncratic shock was 1 
            if raux<=p_u1u1_uu
               idshock(t,i)=1;
            else
               idshock(t,i)=2;
            end
         else                 % if the previous idiosyncratic shock was 2 
            if raux<=p_u1u1_ee
               idshock(t,i)=2;
            else
               idshock(t,i)=1;
            end
         end
      end
   end   
   %{
   if agshock(t-1)==3 && agshock(t)==4 % if the previous agg. shock was unc 
                                      % and the current agg. shock is v.bad
      for i=1:N
         raux=rand;
         if idshock(t-1,i)==1 % if the previous idiosyncratic shock was 1 
            if raux<=p_ub1_uu
               idshock(t,i)=1;
            else
               idshock(t,i)=2;
            end
         else                 % if the previous idiosyncratic shock was 2 
            if raux<=p_ub1_ee
               idshock(t,i)=2;
            else
               idshock(t,i)=1;
            end
         end
      end
   end   
   %}
   
   if agshock(t-1)==3 && agshock(t)==5 % if the previous agg. shock was unc 
                                      % and the current agg. shock is
                                      % v.good
      for i=1:N
         raux=rand;
         if idshock(t-1,i)==1 % if the previous idiosyncratic shock was 1 
            if raux<=p_u1b1_uu
               idshock(t,i)=1;
            else
               idshock(t,i)=2;
            end
         else                 % if the previous idiosyncratic shock was 2 
            if raux<=p_u1b1_ee
               idshock(t,i)=2;
            else
               idshock(t,i)=1;
            end
         end
      end
   end  
   
   if agshock(t-1)==3 && agshock(t)==6 % if the previous agg. shock was unc 
                                      % and the current agg. shock is
                                      % v.good
      for i=1:N
         raux=rand;
         if idshock(t-1,i)==1 % if the previous idiosyncratic shock was 1 
            if raux<=p_u1g1_uu
               idshock(t,i)=1;
            else
               idshock(t,i)=2;
            end
         else                 % if the previous idiosyncratic shock was 2 
            if raux<=p_u1g1_ee
               idshock(t,i)=2;
            else
               idshock(t,i)=1;
            end
         end
      end
   end 
%{
   if agshock(t-1)==4 && agshock(t)==1 % if the previous agg. shock was v.bad 
                                      % and the current agg. shock is bad
      for i=1:N
         raux=rand;
         if idshock(t-1,i)==1 % if the previous idiosyncratic shock was 1 
            if raux<=p_b1b_uu
               idshock(t,i)=1;
            else
               idshock(t,i)=2;
            end
         else                 % if the previous idiosyncratic shock was 2 
            if raux<=p_b1b_ee
               idshock(t,i)=2;
            else
               idshock(t,i)=1;
            end
         end
      end
   end   
   
   if agshock(t-1)==4 && agshock(t)==2 % if the previous agg. shock was v.bad 
                                      % and the current agg. shock is good
      for i=1:N
         raux=rand;
         if idshock(t-1,i)==1 % if the previous idiosyncratic shock was 1 
            if raux<=p_b1g_uu
               idshock(t,i)=1;
            else
               idshock(t,i)=2;
            end
         else                 % if the previous idiosyncratic shock was 2 
            if raux<=p_b1g_ee
               idshock(t,i)=2;
            else
               idshock(t,i)=1;
            end
         end
      end
   end 
   
   if agshock(t-1)==4 && agshock(t)==3 % if the previous agg. shock was v.bad 
                                      % and the current agg. shock is unc
      for i=1:N
         raux=rand;
         if idshock(t-1,i)==1 % if the previous idiosyncratic shock was 1 
            if raux<=p_b1u_uu
               idshock(t,i)=1;
            else
               idshock(t,i)=2;
            end
         else                 % if the previous idiosyncratic shock was 2 
            if raux<=p_b1u_ee
               idshock(t,i)=2;
            else
               idshock(t,i)=1;
            end
         end
      end
   end    
  %} 
   if agshock(t-1)==4 && agshock(t)==4 % if the previous agg. shock was v.bad 
                                      % and the current agg. shock is v.bad
      for i=1:N
         raux=rand;
         if idshock(t-1,i)==1 % if the previous idiosyncratic shock was 1 
            if raux<=p_u2u2_uu
               idshock(t,i)=1;
            else
               idshock(t,i)=2;
            end
         else                 % if the previous idiosyncratic shock was 2 
            if raux<=p_u2u2_ee
               idshock(t,i)=2;
            else
               idshock(t,i)=1;
            end
         end
      end
   end   
   
   if agshock(t-1)==4 && agshock(t)==5 % if the previous agg. shock was v.bad 
                                      % and the current agg. shock is v.good
      for i=1:N
         raux=rand;
         if idshock(t-1,i)==1 % if the previous idiosyncratic shock was 1 
            if raux<=p_u2b1_uu
               idshock(t,i)=1;
            else
               idshock(t,i)=2;
            end
         else                 % if the previous idiosyncratic shock was 2 
            if raux<=p_u2b1_ee
               idshock(t,i)=2;
            else
               idshock(t,i)=1;
            end
         end
      end
   end  
   
  if agshock(t-1)==4 && agshock(t)==6 % if the previous agg. shock was v.bad 
                                      % and the current agg. shock is v.good
      for i=1:N
         raux=rand;
         if idshock(t-1,i)==1 % if the previous idiosyncratic shock was 1 
            if raux<=p_u2g1_uu
               idshock(t,i)=1;
            else
               idshock(t,i)=2;
            end
         else                 % if the previous idiosyncratic shock was 2 
            if raux<=p_u2g1_ee
               idshock(t,i)=2;
            else
               idshock(t,i)=1;
            end
         end
      end
   end  
   
   if agshock(t-1)==5 && agshock(t)==1 % if the previous agg. shock was v.good 
                                      % and the current agg. shock is bad
      for i=1:N
         raux=rand;
         if idshock(t-1,i)==1 % if the previous idiosyncratic shock was 1 
            if raux<=p_b1b_uu
               idshock(t,i)=1;
            else
               idshock(t,i)=2;
            end
         else                 % if the previous idiosyncratic shock was 2 
            if raux<=p_b1b_ee
               idshock(t,i)=2;
            else
               idshock(t,i)=1;
            end
         end
      end
   end    
   
   if agshock(t-1)==6 && agshock(t)==2 % if the previous agg. shock was v.good 
                                      % and the current agg. shock is good
      for i=1:N
         raux=rand;
         if idshock(t-1,i)==1 % if the previous idiosyncratic shock was 1 
            if raux<=p_g1g_uu
               idshock(t,i)=1;
            else
               idshock(t,i)=2;
            end
         else                 % if the previous idiosyncratic shock was 2 
            if raux<=p_g1g_ee
               idshock(t,i)=2;
            else
               idshock(t,i)=1;
            end
         end
      end
   end   
   
   if agshock(t-1)==5 && agshock(t)==5 % if the previous agg. shock was v.good 
                                      % and the current agg. shock is good
      for i=1:N
         raux=rand;
         if idshock(t-1,i)==1 % if the previous idiosyncratic shock was 1 
            if raux<=p_b1b1_uu
               idshock(t,i)=1;
            else
               idshock(t,i)=2;
            end
         else                 % if the previous idiosyncratic shock was 2 
            if raux<=p_b1b1_ee
               idshock(t,i)=2;
            else
               idshock(t,i)=1;
            end
         end
      end
   end    
   
   if agshock(t-1)==5 && agshock(t)==6 % if the previous agg. shock was v.good 
                                      % and the current agg. shock is v.bad
      for i=1:N
         raux=rand;
         if idshock(t-1,i)==1 % if the previous idiosyncratic shock was 1 
            if raux<=p_b1g1_uu
               idshock(t,i)=1;
            else
               idshock(t,i)=2;
            end
         else                 % if the previous idiosyncratic shock was 2 
            if raux<=p_b1g1_ee
               idshock(t,i)=2;
            else
               idshock(t,i)=1;
            end
         end
      end
   end   
   
    if agshock(t-1)==6 && agshock(t)==5 % if the previous agg. shock was v.good 
                                      % and the current agg. shock is v.bad
      for i=1:N
         raux=rand;
         if idshock(t-1,i)==1 % if the previous idiosyncratic shock was 1 
            if raux<=p_g1b1_uu
               idshock(t,i)=1;
            else
               idshock(t,i)=2;
            end
         else                 % if the previous idiosyncratic shock was 2 
            if raux<=p_g1b1_ee
               idshock(t,i)=2;
            else
               idshock(t,i)=1;
            end
         end
      end
   end   
   
   if agshock(t-1)==6 && agshock(t)==6 % if the previous agg. shock was v.good 
                                      % and the current agg. shock is
                                      % v.good
      for i=1:N
         raux=rand;
         if idshock(t-1,i)==1 % if the previous idiosyncratic shock was 1 
            if raux<=p_g1g1_uu
               idshock(t,i)=1;
            else
               idshock(t,i)=2;
            end
         else                 % if the previous idiosyncratic shock was 2 
            if raux<=p_g1g1_ee
               idshock(t,i)=2;
            else
               idshock(t,i)=1;
            end
         end
      end
   end    
end



unemployment=1-(sum(idshock')/N-1); %#ok<UDIM,NASGU>
%% Discount shocks

for i=1:N
    raux=rand;
    if raux<init_dcf(1)
    betashock(1,i)=betas(1);
    elseif raux<init_dcf(1)+init_dcf(2)
        betashock(1,i)=betas(2);
    else
        betashock(1,i)=betas(3);
    end
end

for t=2:T
   for i=1:N
       raux=rand;
       if betashock(t-1,i)==betas(1)
           if raux<pi_beta(1,1)
                betashock(t,i)=betas(1);
           else
               betashock(t,i)=betas(2);
           end
       elseif betashock(t-1,i)==betas(2)
           if raux<pi_beta(2,1)
               betashock(t,i)=betas(1);
           elseif raux<pi_beta(2,1)+pi_beta(2,2)
               betashock(t,i)=betas(2);
           else
               betashock(t,i)=betas(3);
           end
       else
           if raux>pi_beta(3,2)
               betashock(t,i)=betas(3);
           else
               betashock(t,i)=betas(2);
           end
       end
       
   end
end
