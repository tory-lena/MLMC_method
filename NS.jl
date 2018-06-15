# Turbulanece model - Navier stokes equations

function NS()

    nvert = 3      # plotting option (1 for slides, 2 for papers, 3 for full set)
    M     = 8      # refinement cost factor (2^gamma in general MLMC Thm)

    N     = 10000   # samples for convergence tests
    L     = 4      # levels for convergence tests 

    N0    = 1000    # initial number of samples on each MLMC level
    Eps   = [ 0.001, 0.005, 0.01]   # desired accuracies for MLMC calcs

    A = mlmc_test(NS_l, M, N,L, N0,Eps, nvert)
end


function NS_l(l::Int64,N::Int64)

    lam = 1#0.25                              # timestep/spacing^2
    U0=1.;
    nu=.1

    nf  = 2^(l+1)
    hf  = 1/nf                              # grid spacing
    dtf = lam*hf^2                          # timestep

    if l>0
        nc  = Int(nf/2)
        hc  = 1/nc
        dtc = lam*hc^2
    end

    sum1 = zeros(Float64, (1,4))
    sum2 = zeros(Float64, (1,2))

    for N1 = 1:100:N                         # do 100 samples at a time
        N2 = min(100,N-N1+1)
        uf = zeros(nf+1,N2); uf[[1,nf+1],:]=U0*ones(2,N2)
        xf = zeros(nf+1,N2)              # zero initial data
        
        if l==0
            i = 2:nf
            for n = 1:nf^2                       # time-marching
                dWf = sqrt(dtf)*randn(1,N2)
                xf[i,:] = xf[i,:] + dtf*uf[i,:]
                uf[i,:] = uf[i,:] + nu*lam.*(uf[i+1,:]-(2).*uf[i,:]+uf[i-1,:] ) + .25*repmat(dWf,nf-1,1) #+.5*lam*hf*(uf[i+1,:]-uf[i-1,:]) 
            end

            Pf = sum(hf*uf,1)                # output
            Pc = zeros(1,N2)
        else
            uc = zeros(nc+1, N2); uc[[1,nc+1], :]=U0*ones(2,N2)
            xc = zeros(nc+1, N2) 
            for n = 1:nc^2                # time-marching
                dWc = zeros(1,N2)
                for nn = 1:4
                    dWf = sqrt(dtf)*randn(1,N2)
                    dWc = dWc + dWf
                    i = 2:nf
                    xf[i,:] = xf[i,:] + dtf*uf[i,:]
                    uf[i,:] = uf[i,:] + nu*lam.*(uf[i+1,:]-(2).*uf[i,:]+uf[i-1,:] )  + .25*repmat(dWf,nf-1,1) #+.5*lam*hf*(uf[i+1,:]-uf[i-1,:])
                end            
                i = 2:nc
                xc[i,:] = xc[i,:] + dtf*xc[i,:]
                uc[i,:] = uc[i,:] + nu*lam.*(uc[i+1,:]-(2).*uc[i,:]+uc[i-1,:] ) + .25* repmat(dWc,nc-1,1)  #+.5*lam*hc*(uc[i+1,:]-uc[i-1,:]) 
            end
            
            Pf = sum(hf.*uf,1)                # output
            Pc = sum(hc.*uc,1)
            #print(size(uf))
        end
        if l==4 && N1==1 && size(xf)[2]>=10
            figure("MLMC Sampling", figsize=(10, 5))
            subplot(1,2,1)
            plot(xf[2:end-1, 1:10])
            xlabel("Distretized Tube Width")
            title("Particle Position")
            subplot(1,2,2)
            plot(uf[:, 1:10])
            xlabel("Distretized Tube Width")
            title("Velocity Profile")
            #savefig("particle.png")
        end
        
        sum1[1] = sum1[1] + sum(Pf-Pc)
        sum1[2] = sum1[2] + sum((Pf-Pc).^2)
        sum1[3] = sum1[3] + sum((Pf-Pc).^3)
        sum1[4] = sum1[4] + sum((Pf-Pc).^4)
        sum2[1] = sum2[1] + sum(Pf)
        sum2[2] = sum2[2] + sum(Pf.^2)
    end
    
    return  (sum1, sum2)
end