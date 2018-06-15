
# These are similar to the MLMC tests for the original 
# 2008 Operations Research paper, using an Euler-Maruyama 
# discretisation with 4^l timesteps on level l.
#
# The differences are:
# -- the plots do not have the extrapolation results
# -- the top two plots are log_2 rather than log_4
# -- the new MLMC driver is a little different
# -- switch to X_0=100 instead of X_0=1
#

function opre(options::Array{Int64,1})


    N0    = 1000   # initial samples on coarse levels
    Lmin  = 2      # minimum refinement level
    Lmax  = 6      # maximum refinement level

    for option = options
        if (option==1) 
        @printf("\n ---- European call ---- \n")
        N      = 2000000       # samples for convergence tests
        L      = 5             # levels for convergence tests 
        Eps    = [ 0.005, 0.01, 0.02, 0.05, 0.1 ]
      elseif (option==2) 
        @printf("\n ---- Asian call ---- \n")
        N      = 2000000  # samples for convergence tests
        L      = 5        # levels for convergence tests 
        Eps    = [ 0.0005, 0.001, 0.005, 0.01]
      elseif (option==3) 
        @printf("\n ---- lookback call ---- \n")
        N      = 2000000  # samples for convergence tests
        L      = 5        # levels for convergence tests 
        Eps    = [ 0.01, 0.02, 0.05, 0.1, 0.2 ]
      elseif (option==4) 
        @printf("\n ---- digital call ---- \n")
        N      = 3000000  # samples for convergence tests
        L      = 5        # levels for convergence tests 
        Eps    = [ 0.02, 0.05, 0.1, 0.2 ,0.5 ]
      elseif (option==5) 
        @printf("\n ---- Heston model ---- \n")
        N      = 2000000  # samples for convergence tests
        L      = 5        # levels for convergence tests 
        Eps    = [ 0.005, 0.01, 0.02, 0.05, 0.1 ]
      end

      #if (option<5)
      #  filename = ["opre_gbm" num2str(option)]
      #else
      #  filename = "opre_heston"
      #end
      #fp = fopen([filename ".txt"],"w")
        mlmc2_test(opre_l, N,L, N0,Eps,Lmin,Lmax, option, "opre")
      #fclose(fp)

    #
    # print exact analytic value, based on S0=K
    #
        
        T   = 1
        r   = 0.05
        sig = 0.2
        K   = 100

        d1  = (r+0.5*sig^2)*T / (sig*sqrt(T))
        d2  = (r-0.5*sig^2)*T / (sig*sqrt(T))

        if (option==1)
            val = K*( ncf(d1) - exp(-r*T)*ncf(d2) )
            @printf("\n Exact value: %f \n",val)
        elseif (option==3)
            k   = 0.5*sig^2/r
            val = K*( ncf(d1) - ncf(-d1)*k - exp(-r*T)*(ncf(d2) - ncf(d2)*k) )
            @printf("\n Exact value: %f \n",val)
        elseif (option==4)
            val = K*exp(-r*T)*ncf(d2)
            @printf(" Exact value: %f \n\n",val)
          end

    #
    # plot results
    #
      nvert = 3
      #mlmc2_plot(filename, nvert)

    end
end


#-------------------------------------------------------
#
# level l estimator for Operations Research paper
#

function opre_l(l,N, option)

    M = 4

    T   = 1
    r   = 0.05
    sig = 0.2
    K   = 100

    nf = M^l
    nc = nf/M

    hf = T/nf
    hc = T/nc

    sums = zeros(1,6)
    

    for N1 = 1:10000:N
        N2 = min(10000,N-N1+1)

        #
        # GBM model
        #
        if option<5
            X0 = K

            Xf = X0*ones(1,N2)
            Xc = X0*ones(1,N2)

            Af = 0.5*hf*Xf
            Ac = 0.5*hc*Xc

            Mf = X0*ones(1,N2)
            Mc = X0*ones(1,N2)

            if l==0
                dWf = sqrt(hf)*randn(1,N2)
                Xf  = Xf + r*Xf*hf + sig*Xf.*dWf
                Af = Af + 0.5*hf*Xf
                Mf = min.(Mf,Xf)
            else
                for n = 1:nc
                    dWc = zeros(1,N2)
                    
                    for m = 1:M
                        dWf = sqrt(hf)*randn(1,N2)
                        dWc = dWc + dWf
                        Xf  = Xf + r*Xf*hf + sig*Xf.*dWf
                        Af  = Af + hf*Xf
                        Mf  = min.(Mf,Xf)
                    end
                    Xc = Xc + r*Xc*hc + sig*Xc.*dWc
                    Ac = Ac + hc*Xc
                    Mc = min.(Mc,Xc)
                end
                Af = Af - 0.5*hf*Xf
                Ac = Ac - 0.5*hc*Xc
            end
            
            

            if option==1
                Pf = max.(0,Xf-K)
                Pc = max.(0,Xc-K)
            elseif option==2
                Pf = max.(0,Af-K)
                Pc = max.(0,Ac-K)
            elseif option==3
                beta = 0.5826  # special factor for offset correction
                Pf = Xf - Mf*(1-beta*sig*sqrt(hf))
                Pc = Xc - Mc*(1-beta*sig*sqrt(hc))
            elseif option==4
                Pf = K * 0.5*(sign.(Xf-K)+1)
                Pc = K * 0.5*(sign.(Xc-K)+1)
            end
    #
    # Heston model
    #
      else
            X0 = [K 0.04]'
            Xf = X0*ones(1,N2)
            Xc = X0*ones(1,N2)

            if l==0
                dWf = sqrt(hf)*randn(2,N2)
                Xf  = Xf + mu(Xf,hf)*hf + sig_dW(Xf,dWf,hf)

            else
                for n = 1:nc
                    dWc = zeros(2,N2)
                    for m = 1:M
                        dWf = sqrt(hf)*randn(2,N2)
                        dWc = dWc + dWf
                        Xf  = Xf + mu(Xf,hf)*hf + sig_dW(Xf,dWf,hf)
                    end
                    Xc = Xc + mu(Xc,hc)*hc + sig_dW(Xc,dWc,hc)
                end
            end

            Pf = max.(0,Xf(1,:)-K)
            Pc = max.(0,Xc(1,:)-K)
        end

        Pf = exp(-r*T)*Pf
        Pc = exp(-r*T)*Pc

      
        if l==0
            Pc=0
        end

        sums[1] = sums[1] + sum(Pf-Pc)
        sums[2] = sums[2] + sum((Pf-Pc).^2)
        sums[3] = sums[3] + sum((Pf-Pc).^3)
        sums[4] = sums[4] + sum((Pf-Pc).^4)
        sums[5] = sums[5] + sum(Pf)
        sums[6] = sums[6] + sum(Pf.^2)
    end

    cost = N*nf   # cost defined as number of fine timesteps
    
    return sums, cost
end


#AID FUCNTIONS
#--------------------

mu(x,h) = [0.05*x[1,:] ((1-exp(-5.*h))/h)*(0.04-x[2,:])]

#--------------------

function sig_dW(x,dW,h)

    dW[2,:] = -0.5*dW[1,:] + sqrt(0.75)*dW[2,:]

    sigdW = [ sqrt(max(0,x[2,:])).*x[1,:].*dW[1,:]  ...
              exp(-5*h)*0.25*sqrt(max(0,x[2,:])).*dW[2,:] ]
    return sigdW
end

# using SpecialFunctions -------------------------------------

ncf(x) = 0.5*Base.erfc(-x/sqrt(2))