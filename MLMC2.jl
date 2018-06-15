function mlmc2(mlmc2_l, N0::Int64, eps::Float64,Lmin::Int64, Lmax::Int64, alpha0::Float64, beta0::Float64, gamma0::Float64, options=())#, varargin)

#
# check input parameters
#
    if (Lmin<2)
        print["error: needs Lmin >= 2"]
    end

    if (Lmax<Lmin)
        print["error: needs Lmax >= Lmin"]
    end

    if (N0<=0 || eps<=0)
        print["error: needs N0>0, eps>0 \n"]
    end

#
# initialisation
#

    alpha = max(0, alpha0) 
    beta  = max(0, beta0)
    gamma = max(0, gamma0) 

    theta = 0.25 

    L = Lmin 

    Nl = zeros(1, L+1)
    Cl = zeros(1, L+1)
    suml = zeros(2, L+1)
    costl = zeros(1, L+1)
    dNl = N0*ones(1, L+1) 

    while sum(dNl) > 0

#
# update sample sums
#
        for l=0:L
            if dNl[l+1] > 0
                isempty(options)? (sums, cost) = mlmc2_l(l,Int(dNl[l+1])) : (sums, cost) = mlmc2_l(l,Int(dNl[l+1]), options)#, varargin{:}] 
                Nl[l+1]     = Nl[l+1]     + dNl[l+1] 
                suml[1,l+1] = suml[1,l+1] + sums[1] 
                suml[2,l+1] = suml[2,l+1] + sums[2] 
                costl[l+1]  = costl[l+1]  + cost 
            end
        end

#
# compute absolute average, variance and cost
#

        ml = abs.(suml[1,:]./Nl')
        Vl = max.(0, suml[2,:]./Nl' - ml.^2)
        Cl = costl./Nl
        

#
# fix to cope with possible zero values for ml and Vl
# [can happen in some applications when there are few samples]
#
        for l = 3:L+1
            ml[l] = max.(ml[l], 0.5*ml[l-1]/2^alpha)
            Vl[l] = max.(Vl[l], 0.5*Vl[l-1]/2^beta) 
        end
        

#
# use linear regression to estimate alpha, beta, gamma if not given
#
        A = repmat(collect(1:L),1,2).^repmat(1:-1:0,1,L)'

        if alpha0 <= 0
            x = A \ log2.(ml[2:end])
            alpha = max(0.5,-x[1]) 
        end

        if beta0 <= 0
            x = A \ log2.(Vl[2:end])
            beta  = max(0.5,-x[1])
        end

        if gamma0 <= 0
            x = A \ log2.(Cl[2:end])
            gamma = max(0.5,x[1])
        end
        
#
# set optimal number of additional samples
#
        Ns  = ceil.(sqrt.(Vl./Cl')*sum(sqrt.(Vl.*Cl')) / ((1-theta)*eps^2))
        dNl = max.(0, Ns-Nl')
#
# if [almost] converged, estimate remaining error and decide 
# whether a new level is required
        #print(dNl, Nl)
#
        if sum(dNl .> 0.01*Nl ) == 0
            rem = ml[L+1] / (2^alpha - 1)

            if rem > sqrt(theta)*eps
                if L==Lmax
                    print("*** failed to achieve weak convergence *** \n")
                else
                    #print(Vl, Cl, Nl, suml, costl)
                    L       = L+1 
                    Vl=[Vl; Vl[L] / 2^beta]
                    Cl=[Cl Cl[L]*(2^gamma)]
                    Nl=[Nl 0]
                    suml=[suml zeros(2,1) ]
                    costl=[costl 0.]

                    Ns  = ceil.(sqrt.(Vl./Cl')*sum(sqrt.(Vl.*Cl')) / ((1-theta)*eps^2))
                    dNl = max.(0, Ns-Nl')
                end
            end
        end
    end

#
# finally, evaluate multilevel estimator
#
    P = sum(suml[1,:]./Nl') 
    return  P, Nl, Cl
end


function mlmc2_test(mlmc2_l, N::Int64,L::Int64, N0::Int64,Eps::Array{Float64,1},Lmin::Int64,Lmax::Int64, options=(), name::String="testing") #,  varargin)

#
# first, convergence tests
#

    N = 100*ceil(N/100)   # make N a multiple of 100
    @printf("\n l   ave(Pf-Pc)    ave(Pf)   var(Pf-Pc)  var(Pf)   kurtosis    check     cost \n")

    
    del1 = Array{Float64}(1,0)
    del2 = Array{Float64}(1,0)
    var1 = Array{Float64}(1,0)
    var2 = Array{Float64}(1,0)
    kur1 = Array{Float64}(1,0)
    chk1 = Array{Float64}(1,0)
    cost = Array{Float64}(1,0)

    for l = 0:L
    #  disp[sprintf["l = #d",l]]
        sums = 0
        cst  = 0
        for j=1:100
        #RandStream.setGlobalStream[ ...
        #RandStream.create["mrg32k3a","NumStreams",100,"StreamIndices",j]]

            isempty(options)? (sums_j, cst_j) = mlmc2_l(l,Int(N/100)) : (sums_j, cst_j) = mlmc2_l(l,Int(N/100), options) #, varargin{:}]
            sums = sums + sums_j/N
            cst  = cst  + cst_j/N
        end

        if l==0
            kurt = 0.0
        else
            kurt = [sums[4] - 4*sums[3]*sums[1] + 6*sums[2]*sums[1]^2 - 3*sums[1]*sums[1]^3 ] / (sums[2]-sums[1]^2)^2
        end

        cost = [cost cst]
        del1 = [del1 sums[1]]
        del2 = [del2 sums[5]]
        var1 = [var1 sums[2]-sums[1]^2 ]
        var2 = [var2 sums[6]-sums[5]^2 ]
        var2 = max.(var2, 1e-10)  # fix for cases with var=0
        kur1 = [kur1 kurt]

        if l==0
            check = 0
        else
            check = abs(del1[l+1] +del2[l]-del2[l+1])/ (3.*(sqrt(var1[l+1])+sqrt(var2[l])+ sqrt(var2[l+1]) )/sqrt(N))
        end
            chk1 = [chk1 check]

            @printf("%2d  %11.4e %11.4e  %.3e  %.3e  %.2e  %.2e  %.2e \n",l,del1[l+1],del2[l+1],var1[l+1],var2[l+1],kur1[l+1],chk1[l+1],cst)
    end

    #
    # print out a warning if kurtosis or consistency check looks bad
    #

    if  kur1[end] > 100.0 
        @printf("\n WARNING: kurtosis on finest level = %f \n",kur1[end])
        @printf(" indicates mlmc2 correction dominated by a few rare paths \n")
        @printf(" for information on the connection to variance of sample variances,\n")
        @printf(" see http://mathworld.wolfram.com/SampleVarianceDistribution.html\n\n")
    end

    if  maximum(chk1) > 1.0 
        @printf("\n WARNING: maximum consistency error = %f \n", maximum(chk1))
        @printf(" indicates identity E[Pf-Pc] = E[Pf] - E[Pc] not satisfied \n")
        @printf(" to be more certain, re-run mlmc2_test with larger N \n\n")
    end

    #
    # use linear regression to estimate alpha, beta and gamma
    #

    L1 = 2
    L2 = L+1
    range = L1:L2
    
    #print(del1, var1, cost)
    
    @printf("\nestimates of key mlmc2 Theorem parameters based on linear regression: \n")
    pa = polyfit(range,log2.(abs.(del1[range])),1)
    alpha = -pa[1]
    @printf("alpha = %f  (exponent for mlmc2 weak convergence) \n",alpha)
    pb = polyfit(range,log2.(abs.(var1[range])),1)
    beta = -pb[1]
    @printf("beta  = %f  (exponent for mlmc2 variance) \n",beta)
    pg = polyfit(range,log2.(abs.(cost[range])),1)
    gamma = pg[1]
    @printf("gamma = %f  (exponent for mlmc2 cost) \n\n",gamma)
    

    # second, mlmc2 complexity tests

    # reset random number generators

    #reset[RandStream.getGlobalStream]
    srand()
    
    #spmd
    #  RandStream.setGlobalStream[ ...
    #  RandStream.create["mrg32k3a","NumStreams",numlabs,"StreamIndices",labindex]]
    #end

    alpha = max(alpha,0.5)
    beta  = max(beta,0.5)
    theta = 0.25
    
    mlmc2_cost = Array{Float64}(length(Eps),1)
    std_cost = Array{Float64}(length(Eps),1)
    Nls = Array{Array}(length(Eps),)
    ls  = Array{Array}(length(Eps),)
    
    for i = 1:length(Eps)
        eps = Eps[i]
        P, Nl, Cl = mlmc2(mlmc2_l,N0,eps,Lmin,Lmax, alpha,beta,gamma, options)
        mlmc2_cost[i] = sum(Nl.*Cl')
        std_cost[i]  = var2[min(end,length(Nl))]*Cl[end] / ((1.0-theta)*eps^2)
        
        Nls[i] = Nl
        ls[i]  = collect(0.:1.:length(Nl)-1)
        #@printf("%.3e %10.3e  %.3e  %.3e  %7.2f ", eps, P, mlmc2_cost, std_cost, std_cost/mlmc2_cost)
        #@printf("%9d",Nl)
        #@printf("\n")
    end
    L = 0:L
    L = L[:,:]
    #print(Nls, ls)
    plot_my_analysis(3, name, L, var1, var2, del1, del2, chk1, kur1, Nls, ls, Eps, std_cost, mlmc2_cost)

end