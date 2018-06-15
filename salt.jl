function salt()

    N    = 100000 # samples for convergence tests
    L    = 6     # levels for convergence tests 

    N0   = 200    # initial samples on each level
    Eps  = [0.00005, 0.0001, 0.00025, 0.0005] # target accuracies
    Lmin = 2      # minimum refinement level
    Lmax = 10     # maximum refinement level

    X0  = 5 #K    # initial position
    Y0  = 1    # initial velocity for particle with mass
    Z0  = 1
    T   = 1       # time interval

    a   =  0.25  # drift
    b   =  0.4  # volatility
    
    mlmc2_test(salt_l, N,L, N0,Eps,Lmin,Lmax,  (X0, Y0, Z0, T,a,b), "salt")
    
end


#-------------------------------------------------------
#
# level l estimator for reflected simulation with mass
#

function salt_l(l::Int64,N::Int64, options::Tuple)

    (X0,Y0, Z0, T,a,b)=options

    M  = 2

    nf = M^l
    nc = nf/M

    hf = T/nf
    hc = T/nc

    sums= zeros(1,6)

    for N1 = 1:10000:N
        N2 = min(10000,N-N1+1)

        Xf  = X0*ones(1,N2)
        Xc  = X0*ones(1,N2)
        Yf  = Y0*ones(1,N2)
        Yc  = Y0*ones(1,N2)
        Zf  = Z0*ones(1,N2)
        Zc  = Z0*ones(1,N2)
        Pf  = zeros(1,N2)
        Pc  = zeros(1,N2)

        if l==0
            dWf = sqrt(hf)*randn(1,N2)
            Xf = Xf + (- Yf.^2 - Zf.^2 - a*Xf)*hf + a*dWf 
            Yf = Yf + (Xf.*Yf - b*Xf.*Zf - Yf)*hf + dWf
            Zf = Zf + (b*Xf.*Yf + Xf.*Zf - Zf)*hf
                    
            #Tf = 25. - .063634*X
            Pf = Pf + (.00166 +.00022*(Yf.^2+Zf.^2))*hf

        else
            for n = 1:nc
                dWc = zeros(1,N2)

                for m = 1:M
                    dWf = sqrt(hf)*randn(1,N2)
                    dWc = dWc + dWf

                    Xf = Xf + (- Yf.^2 - Zf.^2 - a*Xf)*hf + a*dWf 
                    Yf = Yf + (Xf.*Yf - b*Xf.*Zf - Yf)*hf + dWf
                    Zf = Zf + (b*Xf.*Yf + Xf.*Zf - Zf)*hf
                    
                    #Tf = 25. - .063634*X
                    Pf = Pf + (.00166 +.00022*(Yf.^2+Zf.^2))*hf
                end
                
                Xc = Xc + hc*(- Yc.^2 - Zc.^2 - a*Xc) + a*dWc 
                Yc = Yc + hc*(Xc.*Yc - b*Xc.*Zc - Yc) + dWc
                Zc = Zc + hc*(b*Xc.*Yc + Xc.*Zc - Zc)
                    
                #Tc = 25. - .063634*X
                Pc = Pc + (.00166 +.00022*(Yc.^2+Zc.^2))*hc
            end
        end

        #  Pf = Xf   # uncomment these lines to switch to final position
        #  if (l>0)
        #    Pc = Xc
        #  end
    
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