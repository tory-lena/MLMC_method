#
# This tests the use of mlmc22 for a reflected diffusion
# in the interval [0,1] with reflections at either end
#

function reflected()

    N    = 100000 # samples for convergence tests
    L    = 10     # levels for convergence tests 

    N0   = 200    # initial samples on each level
    Eps  = [ 0.0002, 0.0005, 0.001, 0.002, 0.005 ] # target accuracies
    Lmin = 2      # minimum refinement level
    Lmax = 10     # maximum refinement level

    for mass = 1:1 #0:1
        for p = 1:2

            X0  =  0.2    # initial position
            U0  = -0.2    # initial velocity for particle with mass
            T   = 1       # time interval

            if p==1
                a   =  0.0  # drift
                b   =  0.5  # volatility
                bp  =  0.0  # coefficient for linear variation in volatility
            else
                a   = -0.2  # drift
                b   =  0.5  # volatility
                bp  =  0.5  # coefficient for linear variation in volatility
            end

            if mass==0
                #filename = ["massless_" num2str(p)]
                #fp = fopen([filename ".txt"],"w")
                mlmc2_test(massless_l, N,L, N0,Eps,Lmin,Lmax, (X0, T,a,b,bp), "refl_massless$p")
                #fclose(fp)
            else
                #filename = ["massive_" num2str(p)]
                #fp = fopen([filename ".txt"],"w")
                mlmc2_test(massive_l, N,L, N0,Eps,Lmin,Lmax,  (X0, U0, T,a,b,bp), "refl_massive$p")
                #fclose(fp)
            end

            nvert = 3
            #mlmc22_plot(filename, nvert)
        end
    end
end


# level l estimator for reflected simulation without mass
#
# the drift is a,  the volatility is b + bp*x
#

function massless_l(l::Int64,N::Int64, options::Tuple)
    
    (X0, T, a, b, bp)=options

    M  = 2

    nf = M^l
    nc = nf/M

    hf = T/nf
    hc = T/nc

    sums = zeros(1,6)

    for N1 = 1:10000:N
        N2 = min(10000,N-N1+1)

        Xf  = X0*ones(1,N2)
        Xc  = X0*ones(1,N2)
        Sf  = ones(1,N2)
        Sc  = ones(1,N2)
        Pf  = zeros(1,N2)
        Pc  = zeros(1,N2)

        if l==0
            dWf = sqrt(hf)*randn(1,N2)
            Xf  = Xf + a*hf + (b+bp*Xf).*Sf.*dWf   + 0.5*bp*(b+bp*Xf).*(dWf.^2-hf)

            Xf  = mod.(Xf,2)       # modulo shift into [0,2] range
            ind = find(x-> x>1, Xf)      # then reflect if needed
            Xf[ind] = 2 - Xf[ind]
            Sf[ind] =   - Sf[ind]

            Pf = Pf + hf*Xf

        else
            for n = 1:nc
                dWc = zeros(1,N2)

                for m = 1:M
                    dWf = sqrt(hf)*randn(1,N2)
                    dWc = dWc + dWf

                    Xf  = Xf  + a*hf + (b+bp*Xf).*Sf.*dWf   + 0.5*bp*(b+bp*Xf).*(dWf.^2-hf)

                    Xf  = mod.(Xf,2)       # modulo shift into [0,2] range
                    ind = find(x-> x>1, Xf)     # then reflect if needed
                    Xf[ind] = 2 - Xf[ind]
                    Sf[ind] =   - Sf[ind]
                    Pf = Pf + hf*Xf
                    
                    #print(size(Pf))
                    #break
                end
 
                Xc  = Xc + a*hc + (b+bp*Xc).*Sc.*dWc + 0.5*bp*(b+bp*Xc).*(dWc.^2-hc)
                #print(Xc)

                Xc  = mod.(Xc,2)       # module shift into [0,2] range
                ind = find(x-> x>1, Xc)     # then reflect if needed
                Xc[ind] = 2 - Xc[ind]
                Sc[ind] =   - Sc[ind]

                Pc = Pc + hc*Xc
                #print(size(Pc))
            end
        end

        # Pf = Xf   # uncomment these lines to switch to final position
        # if (l>0)
        #   Pc = Xc
        # end
        #print(Pc)
    

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


#-------------------------------------------------------
#
# level l estimator for reflected simulation with mass
#

function massive_l(l::Int64,N::Int64, options::Tuple)

    (X0,U0,T,a,b,bp)=options

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
        Uf  = U0*ones(1,N2)
        Uc  = U0*ones(1,N2)
        Sf  = ones(1,N2)
        Sc  = ones(1,N2)
        Pf  = zeros(1,N2)
        Pc  = zeros(1,N2)

        if l==0
            dWf = sqrt(hf)*randn(1,N2)
            Xf  = Xf + Uf*hf
            Uf  = Uf + a*hf + (b+bp*Xf).*Sf.*dWf

            Xf  = mod.(Xf,2)       # modulo shift into [0,2] range
            ind = find(x-> x>1, Xf)       # then reflect if needed
            Xf[ind] = 2 - Xf[ind]
            Uf[ind] =   - Uf[ind]
            Sf[ind] =   - Sf[ind]

            Pf = Pf + hf*Xf

        else
            for n = 1:nc
                dWc = zeros(1,N2)

                for m = 1:M
                    dWf = sqrt(hf)*randn(1,N2)
                    dWc = dWc + dWf

                    Xf  = Xf + Uf*hf
                    Uf  = Uf + a*hf + (b+bp*Xf).*Sf.*dWf

                    Xf  = mod.(Xf,2)       # modulo shift into [0,2] range
                    ind = find(x-> x>1, Xf)       # then reflect if needed
                    Xf[ind] = 2 - Xf[ind]
                    Uf[ind] =   - Uf[ind]
                    Sf[ind] =   - Sf[ind]

                    Pf = Pf + hf*Xf
                end
                
                Xc  = Xc + Uc*hc
                Uc  = Uc + a*hc + (b+bp*Xc).*Sc.*dWc

                Xc  = mod.(Xc,2)       # module shift into [0,2] range
                ind = find(x-> x>1, Xc)       # then reflect if needed
                Xc[ind] = 2 - Xc[ind]
                Uc[ind] =   - Uc[ind]
                Sc[ind] =   - Sc[ind]

                Pc = Pc + hc*Xc
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