# This tests the use of multilevel Monte Carlo for a simple parabolic PDE
#
# SPDE: du = d^2u/dx^2 dt + 10 dW
# u(0,t)=0, u(1,t)=0, u(x,0) = 0
#
# the output is P = \int_0^1 u^2(x,0.25) dx

function para()

    nvert = 3      # plotting option (1 for slides, 2 for papers, 3 for full set)
    M     = 8      # refinement cost factor (2^gamma in general MLMC Thm)

    N     = 1000   # samples for convergence tests
    L     = 6      # levels for convergence tests 

    N0    = 100    # initial number of samples on each MLMC level
    Eps   = [ 0.005, 0.01, 0.02, 0.05, 0.1 ]   # desired accuracies for MLMC calcs

    A = mlmc_test(para_l, M, N,L, N0,Eps, nvert)
end


function para_l(l::Int64,N::Int64)

    lam = 0.25                              # timestep/spacing^2

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
        uf = zeros(nf+1,N2)                    # zero initial data
        if l==0
            i = 2:nf
            for n = 1:nf^2                       # time-marching
                dWf = sqrt(dtf)*randn(1,N2)
                uf[i,:] = uf[i,:] + lam.*(uf[i+1,:]-(2).*uf[i,:]+uf[i-1,:] ) + 10 .* repmat(dWf,nf-1,1)
            end

            Pf = sum(hf*uf.^2,1)                # output
            Pc = zeros(1,N2)
        else
            uc = zeros(nc+1,N2)
            for n = 1:nc^2                # time-marching
                dWc = zeros(1,N2)
                for nn = 1:4
                    dWf = sqrt(dtf)*randn(1,N2)
                    dWc = dWc + dWf
                    i = 2:nf
                    uf[i,:] = uf[i,:] + lam.*(uf[i+1,:]-(2).*uf[i,:]+uf[i-1,:] ) + 10 .* repmat(dWf,nf-1,1)
                end            
                i = 2:nc
                uc[i,:] = uc[i,:] + lam.*(uc[i+1,:]-(2).*uc[i,:]+uc[i-1,:] ) + 10 .* repmat(dWc,nc-1,1)
            end
            
            Pf = sum(hf.*uf.^2,1)                # output
            Pc = sum(hc.*uc.^2,1)
            #print(size(uf))
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