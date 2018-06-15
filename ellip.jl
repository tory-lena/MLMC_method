# Solves for d/dx(c(x) du/dx) = -50 Z^2, 	u(0)=u(1)=0
#						Z - N(0,1) 


function ellip()

# call mlmc_test to perform MLMC tests

    nvert = 3 # plotting option (1 for slides, 2 for papers, 3 for full set)
    M     = 2 # refinement cost factor (2^gamma in general MLMC Thm)

    N     = 10000 # samples for convergence tests
    L     = 6 # levels for convergence tests 

    N0    = 200 # initial number of samples on first 3 MLMC levels
    E   = [0.0005, 0.001, 0.005, 0.01]#, 0.02, 0.05, 0.1 ] # desired accuracies for MLMC calcs

    #figs = 
    A = mlmc_test(ellip_l, M, N, L, N0, E, nvert)

end

# level l estimator for elliptic solver

function ellip_l(l::Int64,N::Int64)

# global

    nf  = 2^(l+1)
    hf  = 1/nf
    cf  = ones(Float64, (nf,1))
    A0f = hf^(-2)*spdiagm((cf[2:end-1], -cf[2:end]-cf[1:end-1], cf[2:end-1]),(-1,0,1),nf-1,nf-1)
    cf  = ((1:nf)'-0.5)*hf
    A1f = hf^(-2)*spdiagm((cf[2:end-1], -cf[2:end]-cf[1:end-1], cf[2:end-1]),(-1,0,1),nf-1,nf-1)
    cf  = ones(Float64, (nf-1,1))

    if l>0
        nc = Int(nf/2)
        hc = 1/nc
        cc  = ones(Float64, (nc,1))
        A0c = hc^(-2)*spdiagm((cc[2:end-1], -cc[2:end]-cc[1:end-1], cc[2:end-1]),(-1,0,1),nc-1,nc-1)
        cc  = ((1:nc)'-0.5)*hc
        A1c = hc^(-2)*spdiagm((cc[2:end-1], -cc[2:end]-cc[1:end-1], cc[2:end-1]),(-1,0,1),nc-1,nc-1)
        cc  = ones(Float64, (nc-1,1))
    end

    sum1 = zeros(Float64, (1,4))
    sum2 = zeros(Float64, (1,2))

    for N1 = 1:N         # compute samples 1 at a time
        U  = rand(1,1)
        Z  = randn(1,1)
        
        uf = - (A0f+U.*A1f) \ ((50*Z^2).*cf)
        Pf = sum(hf.*uf)
        if l==0
            Pc = 0
        else
            uc = - (A0c+U.*A1c) \ ((50*Z^2).*cc)
            Pc = sum(hc.*uc)
        end
        sum1 = sum1' + ((Pf-Pc)*ones(4,1)).^(1:4)
        sum1 = sum1'
        sum2 = sum2' + ((Pf)*ones(2,1)).^(1:2)
        sum2 = sum2'
    end
    return  (sum1, sum2)
end