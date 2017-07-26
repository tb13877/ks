function [kmts,kcross,savings]  = AGGREGATE_death(T,idshock,agshock,betashock,km_max,km_min,...
    kprime,km,k,epsilon2,k_min,k_max,kcross,a2,N,deathshock,redist,IHT,p_death,betas)

kmts=zeros(T,1);         % a time series of the mean of capital
savings=zeros(T,N);

for t=1:T
   kcross=kcross.*deathshock(t,:); 
   if t>1
       kcross=kcross+redist*IHT*p_death*kmts(t-1);
   end
   savings(t,:)=kcross;
   kmts(t)=mean(kcross); 
   
   kmts(t)=kmts(t)*(kmts(t)>=km_min)*(kmts(t)<=km_max)+km_min*(kmts(t)<km_min)...
       +km_max*(kmts(t)>km_max); % restrict kmts

   kprimet4=interpn(k,km,a2,epsilon2,betas,kprime,k, kmts(t),agshock(t),epsilon2,betas,'spline');
      
   kprimet=squeeze(kprimet4); % remove unneeded dimensions
                              
   kcrossn=interpn(k,epsilon2,betas,kprimet,kcross,idshock(t,:),betashock(t,:),'spline'); 
                                 
   kcrossn=kcrossn.*(kcrossn>=k_min).*(kcrossn<=k_max)+k_min*(kcrossn<k_min)...
       +k_max*(kcrossn>k_max); % restrict kcross
                              
   kcross=kcrossn;
   
end