# Solves for Population Model by Dawson:
#	du/dt (t,x) = nu d^2 u/dx^2 + alpha *sqrt(u(t,x)) dW/dt(t,x)
# 	with W -> Brownian Motion


function pop()

    nvert = 3      # plotting option (1 for slides, 2 for papers, 3 for full set)
    M     = 8      # refinement cost factor (2^gamma in general MLMC Thm)

    N     = 1000   # samples for convergence tests
    L     = 6      # levels for convergence tests 

    N0    = 100    # initial number of samples on each MLMC level
    Eps   = [100., 50., 10., 5., 1.]   # desired accuracies for MLMC calcs

    A = mlmc_test(pop_l, M, N,L, N0,Eps, nvert)
end


function pop_l(l::Int64,N::Int64)

    a = .1299
    nu = .03
    lam = 1                           # timestep/spacing^2

    nf  = 2^(l+1)
    hf  = 1/nf                              # grid spacing
    kf = lam*hf^2                          # timestep

    if l>0
        nc  = Int(nf/2)
        hc  = 1/nc
        kc = lam*hc^2
    end

    sum1 = zeros(Float64, (1,4))
    sum2 = zeros(Float64, (1,2))

    for N1 = 1:100:N                         # do 100 samples at a time
        N2 = min(100,N-N1+1)
        uf = 100*ones(nf,N2)                    # zero initial data
        if l==0
            i = collect(1:nf)
            for n = 1:nf^2                       # time-marching
                dWf = sqrt(kf)*randn(1,N2)
                uf[i,:] = uf[i,:] + (nu*lam)*(uf[[nf; i[1:end-1]],:]-2.*uf[i,:]+uf[[i[2:end]; 1],:]) + (a*sqrt.(uf[i,:])).* repmat(dWf,nf,1)
            end

            Pf = sum(hf*uf.^2,1)                # output
            Pc = zeros(1,N2)
        else
            uc = 100*ones(nc+1,N2)
            for n = 1:nc^2                # time-marching
                dWc = zeros(1,N2)
                for nn = 1:4
                    dWf = sqrt(kf)*randn(1,N2)
                    dWc = dWc + dWf
                    i = collect(1:nf)
                    uf[i,:] = uf[i,:] + (nu*lam)*(uf[[nf; i[1:end-1]],:]-2.*uf[i,:]+uf[[i[2:end]; 1],:]) + (a*sqrt.(uf[i,:])).* repmat(dWf,nf,1)
                    end            
                i = collect(1:nc)
                uc[i,:] = uc[i,:] + (nu*lam)*(uc[[nc; i[1:end-1]],:]-2.*uc[i,:]+uc[[i[2:end]; 1],:]) + (a*sqrt.(uc[i,:])).* repmat(dWc,nc,1)
            end
            
            #print(uf)
            
            Pf = sum(hf.*uf,1)                # output
            Pc = sum(hc.*uc,1)
            #print(Pf)
        end
        
        sum1[1] = sum1[1] + sum(Pf-Pc)
        sum1[2] = sum1[2] + sum((Pf-Pc).^2)
        sum1[3] = sum1[3] + sum((Pf-Pc).^3)
        sum1[4] = sum1[4] + sum((Pf-Pc).^4)
        sum2[1] = sum2[1] + sum(Pf)
        sum2[2] = sum2[2] + sum(Pf.^2)
    end
    #print(size(sum1), size(sum2))
    return  (sum1, sum2)
end