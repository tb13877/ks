function [V]=VFI(V,Vp,dif_v,util,tol,betas,p_death,prob_e,prob_u,...
    nstates_ag,nstates_id,ngridk,ngridkm,k,km,a2,epsilon2,kmprime,kprime)
% Value function iteration
    while dif_v>tol  
        tic
        p=1; %#ok<NASGU>
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
        dif_v=max(max(max(max(max(abs(V-Vp))))))  %#ok<NOPRT>
        V=Vp;
        toc
    end