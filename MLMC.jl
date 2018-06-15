

function mlmc(N_0::Int64, eps::Float64, mlmc_l, alpha_0::Float64, beta_0::Float64, gamma::Float64)
    alpha = max(0., alpha_0)
    beta = max(0., beta_0)

    L = 2
    N_l = zeros(Float64, (1,3))
    sum_l = zeros(Float64, (2,3))
    dN_l = N_0.*ones(Int64, (1,3))

    while sum(dN_l) > 0

# update sample sums
        
        for l=0:L
            if dN_l[l+1] > 0
                sums         = mlmc_l(l,dN_l[l+1])[1]
                N_l[l+1]     = N_l[l+1] + dN_l[l+1]
                sum_l[1,l+1] = sum_l[1,l+1] + sums[1]
                sum_l[2,l+1] = sum_l[2,l+1] + sums[2]
            end
        end

# compute absolute average and variance
        
        m_l = abs.(sum_l[1,:]'./N_l)
        V_l = max.(0., sum_l[2,:]'./N_l - m_l.^2)


# fix to cope with possible zero values for ml and Vl
# (can happen in some applications when there are few samples)

        for l = 3:L+1
            m_l[l] = max(m_l[l], 0.5*m_l[l-1]/2^alpha)
            V_l[l] = max(V_l[l], 0.5*V_l[l-1]/2^beta)
        end

# use linear regression to estimate alpha, beta if not given

        if alpha_0 <= 0
            A = repmat((1:L)',1,2).^repmat(1:-1:0,L,1)
            x = A \ log2.(m_l[2:end])
            alpha = max(0.5,-x[1])
        end

        if beta_0 <= 0
            A = repmat((1:L)',1,2).^repmat(1:-1:0,L,1)
            x = A \ log2(V_l[2:end])
            beta = max(0.5,-x[1])
        end

# set optimal number of additional samples

        C_l  = 2.^(gamma*(0:L))
        N_s  = Int.(ceil.(2 * sqrt.(V_l'./C_l) * sum(sqrt.(V_l'.*C_l)) ./ eps^2))'
        dN_l = Int.(max.(0, N_s-N_l))

# if (almost) converged, estimate remaining error and decide 
# whether a new level is required

# == true/ any(.>) -> false
        
        if all(dN_l .<= 0.01*N_l) 
            range = [-2 -1 0]
            rem = maximum(m_l[L+1+range].*(2).^(alpha*range)) / (2^alpha - 1)
            
            if rem > eps/sqrt(2)
                L       = L+1
                V_l = [V_l V_l[L] / 2^beta]
                N_l = [N_l 0]
                sum_l = [sum_l zeros(size(sum_l)[1], 1)]
                if size(sum_l)[1]<4
                    sum_l=[sum_l; zeros(4-size(sum_l)[1], size(sum_l)[2])]
                end
                
                C_l  = 2.^(gamma*(0:L))
                N_s  = Int.(ceil.(2 * sqrt.(V_l'./C_l) * sum(sqrt.(V_l'.*C_l)) / eps^2))'
                dN_l = Int.(max.(0, N_s-N_l))
            end
        end
    end


# finally, evaluate multilevel estimator
    #print(sum_l, N_l)

    P = sum(sum_l[1,find(N_l)]./N_l[find(N_l)])
    return (P, N_l)
end





function mlmc_test(mlmc_fn, M::Int64, N::Int64, L::Int64, N_0::Int64, Eps::Array{Float64,1}, nvert::Int64)

# first, convergence tests
    #rng('default');    # reset random number generator

    del1 = Array{Float64}(1,0)
    del2 = Array{Float64}(1,0)
    var1 = Array{Float64}(1,0)
    var2 = Array{Float64}(1,0)
    kur1 = Array{Float64}(1,0)
    chk1 = Array{Float64}(1,0)
    cost = Array{Float64}(1,0)

    L = 0:L
    L = L[:,:]
    #print(L)

    for l in L
        @printf("l = %d\n",l)
        tic()
        (sum1, sum2) = mlmc_fn(l,N)
        cost = [cost toq()]
        sum1 = sum1/N
        sum2 = sum2/N
        kurt = (sum1[4] - 4*sum1[3]*sum1[1] + 6*sum1[2]*sum1[1]^2 - 3*sum1[1]^4) / (sum1[2]-sum1[1]^2)^2
        del1 = [del1 sum1[1]]
        del2 = [del2 sum2[1]]
        var1 = [var1 sum1[2]-sum1[1]^2 ]
        var2 = [var2 sum2[2]-sum2[1]^2 ]
        var2 = max.(var2, 1e-12)  # fix for cases with var=0
        kur1 = [kur1 kurt]
        #print(kurt)

        if l==0
            check = 0
        else
            check = abs(del1[l+1]+del2[l]-del2[l+1])/(3.0*(sqrt(var1[l+1])+sqrt(var2[l])+sqrt(var2[l+1]))/sqrt(N))
        end
        chk1 = [chk1 check]
        #print(check)
    end

# use linear regression to estimate alpha, beta and gamma

    range = Int(max(2,floor(0.4*length(L)))):length(L)
    @printf("\nestimates of key MLMC Theorem parameters based on linear regression: \n")
    pa = polyfit(L[range],log2.(abs.(del1[range])),1)
    alpha = -pa[1]
    @printf("alpha = %f  (exponent for MLMC weak convergence) \n",alpha)
    pb = polyfit(L[range],log2.(abs.(var1[range])),1)
    beta = -pb[1]
    @printf("beta  = %f  (exponent for MLMC variance) \n",beta)
    gamma = log2(cost[end]/cost[end-1])
    @printf("gamma = %f  (exponent for MLMC cost) \n\n",gamma)

    if maximum(chk1) > 1
        @printf("WARNING: maximum consistency error = %f \n",maximum(chk1))
        @printf("indicates identity Eps[Pf-Pc] = Eps[Pf] - Eps[Pc] not satisfied \n\n")
    end

    if kur1[end] > 100
        @printf("WARNING: kurtosis on finest level = %f \n",kur1[end])
        @printf("indicates MLMC correction dominated by a few rare paths; \n")
        @printf("for information on the connection to variance of sample variances, \n")
        @printf("see http://mathworld.wolfram.com/SampleVarianceDistribution.html\n\n")
    end
    
# second, mlmc complexity tests

    #rng('default');    % reset random number generator
    Nls = Array{Array}(length(Eps),)#zeros(6,length(E))
    ls  = Array{Array}(length(Eps),)#zeros(6,length(E))
    
    maxl = 0
    mlmc_cost = Array{Float64}(length(Eps),1)
    std_cost = Array{Float64}(length(Eps),1)
    
    for i = 1:length(Eps)
        eps = Eps[i]
        #@printf("eps = %f \n",eps)
        gamma = log2(M)
        (P, Nl) = mlmc(N_0,eps,mlmc_fn,alpha,beta,gamma)
        #@printf("P = %f \n", P)
        #@printf("Nl = %f \n", Nl)
        #P=sol_mlmc[1]; Nl=sol_mlmc[1]; 
        l = length(Nl)-1
        
        maxl = max(l,maxl)
        mlmc_cost[i] = (1+1/M)*sum(Nl.*M.^(0:l))
        std_cost[i]  = sum((2*var2[end]/eps^2).*M.^(0:l))

        # fprintf(' mlmc_cost = %d, std_cost = %d, savings = %f \n', mlmc_cost(i), std_cost(i), std_cost(i)/mlmc_cost(i))
        Nls[i] = Nl
        ls[i]  = collect(0.:1.:l)
    end

    print("  ")

    #for i = 1:length(E)
    #    Nls[end:maxl,i] = Nls[end,i]
    #    ls[end:maxl,i]  = ls[end,i]
    #end

# plot figures
    
    if nvert == 2 || nvert == 3
        plot_my_analysis(nvert, L, var1, var2, del1, del2, chk1, kur1, Nls, ls, Eps, std_cost, mlmc_cost)
    end
    #return (mlmc_cost, std_cost)
    #return 0 #figs
end






function plot_my_analysis(nvert::Int64, name::String, L::Array{Int64,2}, var1::Array{Float64,2}, var2::Array{Float64,2}, del1::Array{Float64,2}, del2::Array{Float64,2}, chk1::Array{Float64,2}, kur1::Array{Float64,2}, Nls::Array{Array,1}, ls::Array{Array,1}, E::Array{Float64,1}, std_cost::Array{Float64,2}, mlmc_cost::Array{Float64,2})

    ## graph 1 
    #fig=figure("Results", figsize=(12,12))
    
    fig = figure("variance_mean",figsize=(5,4))
    #subplot(2,2,1)
    
    p = plot(L[:,:],log2.(var2'),color="blue",linestyle="-",marker="o")
    plot(L[2:end,:],log2.(var1'[2:end]), color="blue",linestyle="--",marker="o") # Plot a basic line
    legend(["P_l", "P_l- P_{l-1}"])
    ax = gca()
    title("Log-Variance & Log-Mean")

    xlabel("Level L")
    font1 = Dict("color"=>"blue")
    ylabel("log_2 variance",fontdict=font1)
    setp(ax[:get_yticklabels](),color="blue") # Y Axis font formatting
    
    #  Other Axes  #

    new_position = [0.06;0.06;0.77;0.91] # Position Method 2
    ax[:set_position](new_position) # Position Method 2: Change the size and position of the axis
    #fig[:subplots_adjust](right=0.85) # Position Method 1

    ax2 = ax[:twinx]() # Create another axis on top of the current axis
    font2 = Dict("color"=>"purple")
    ylabel("log_2 |mean|",fontdict=font2)
    p = plot_date(L[:,:],log2.(abs.(del2')),color="purple",linestyle="-",marker="o")
    plot(L[2:end,:],log2.(abs.(del1'[2:end])),color="purple",linestyle="--",marker="o") # Plot a basic line
    #legend(["P_l", "P_l- P_{l-1}"])
    ax2[:set_position](new_position) # Position Method 2
    setp(ax2[:get_yticklabels](),color="purple") # Y Axis font formatting

    fig[:canvas][:draw]()
    #png(string(name,"1"))
    
        
    fig2 = figure("consistency_kurtosis",figsize=(5,4))
    
    p = plot(L[2:end,:]-1e-9,chk1'[2:end],linestyle="-",marker="o",label="First") # Plot a basic line
    ax = gca()
    title("Consistency Check & Kurtosis")

    xlabel("Level L")
    font1 = Dict("color"=>"blue")
    ylabel("Consistency Check",fontdict=font1)
    setp(ax[:get_yticklabels](),color="blue") # Y Axis font formatting

    #  Other Axes  #

    new_position = [0.06;0.06;0.77;0.91] # Position Method 2
    ax[:set_position](new_position) # Position Method 2: Change the size and position of the axis
    #fig[:subplots_adjust](right=0.85) # Position Method 1

    ax2 = ax[:twinx]() # Create another axis on top of the current axis
    font2 = Dict("color"=>"purple")
    ylabel("Kurtosis",fontdict=font2)
    p = plot_date(L[2:end,:]-1e-9,kur1'[2:end],color="purple",linestyle="-",marker="o",label="Second") # Plot a basic line
    ax2[:set_position](new_position) # Position Method 2
    setp(ax2[:get_yticklabels](),color="purple") # Y Axis font formatting

    fig2[:canvas][:draw]() 
    # Update the figure
    # gcf() # Needed for IJulia to plot inline
    #savefig(fig2, string(name,"2.png"))
    #savefig("multi-axis.")
    
    figure("N_l", (5,4))
    for i=1:length(ls)
        semilogy(ls[i], collect(Nls[i]'[:]), "--*")
    end
    xlabel("Level L")
    ylabel("N_l")
    xlim((2,length(L)-1))
    grid(true)
    title("Nr of Samples per Level")
    #legend(["$E[i]" for i=1:length(E)])
    #savefig(string(name,"3.png"))
    #set(0,'DefaultAxesLineStyleOrder','-*|--*')

    figure("E", (5,4))
    loglog(E,(E.^2).*std_cost)
    loglog(E,(E.^2).*mlmc_cost)
    legend(["Std_MC", "MLMC"])
    title("Cost")
    xlabel("Accuracy eps")
    ylabel("eps^2 Cost")
    grid(true)
    #axis([0.005, 0.1, 50, 1e4])
    #legend(loc="best")
    #savefig(string(name,"4.png"))
end




function plot_my_analysis(nvert::Int64, L::Array{Int64,2}, var1::Array{Float64,2}, var2::Array{Float64,2}, del1::Array{Float64,2}, del2::Array{Float64,2}, chk1::Array{Float64,2}, kur1::Array{Float64,2}, Nls::Array{Array,1}, ls::Array{Array,1}, E::Array{Float64,1}, std_cost::Array{Float64,2}, mlmc_cost::Array{Float64,2})

    ## graph 1 
    
    fig = figure("variance_mean",figsize=(5,4))
    p = plot(L[:,:],log2.(var2'),color="blue",linestyle="-",marker="o")
    plot(L[2:end,:],log2.(var1'[2:end]), color="blue",linestyle="--",marker="o") # Plot a basic line
    legend(["P_l", "P_l- P_{l-1}"])
    ax = gca()
    title("Log-Variance & Log-Mean")

    xlabel("Level L")
    font1 = Dict("color"=>"blue")
    ylabel("log_2 variance",fontdict=font1)
    setp(ax[:get_yticklabels](),color="blue") # Y Axis font formatting
    
    #  Other Axes  #

    new_position = [0.06;0.06;0.77;0.91] # Position Method 2
    ax[:set_position](new_position) # Position Method 2: Change the size and position of the axis
    #fig[:subplots_adjust](right=0.85) # Position Method 1

    ax2 = ax[:twinx]() # Create another axis on top of the current axis
    font2 = Dict("color"=>"purple")
    ylabel("log_2 |mean|",fontdict=font2)
    p = plot_date(L[:,:],log2.(abs.(del2')),color="purple",linestyle="-",marker="o")
    plot(L[2:end,:],log2.(abs.(del1'[2:end])),color="purple",linestyle="--",marker="o") # Plot a basic line
    #legend(["P_l", "P_l- P_{l-1}"])
    ax2[:set_position](new_position) # Position Method 2
    setp(ax2[:get_yticklabels](),color="purple") # Y Axis font formatting

    fig[:canvas][:draw]()
    
        
    fig2 = figure("consistency_kurtosis",figsize=(5,4))
    p = plot(L[2:end,:]-1e-9,chk1'[2:end],linestyle="-",marker="o",label="First") # Plot a basic line
    ax = gca()
    title("Consistency Check & Kurtosis")

    xlabel("Level L")
    font1 = Dict("color"=>"blue")
    ylabel("Consistency Check",fontdict=font1)
    setp(ax[:get_yticklabels](),color="blue") # Y Axis font formatting

    #  Other Axes  #

    new_position = [0.06;0.06;0.77;0.91] # Position Method 2
    ax[:set_position](new_position) # Position Method 2: Change the size and position of the axis
    #fig[:subplots_adjust](right=0.85) # Position Method 1

    ax2 = ax[:twinx]() # Create another axis on top of the current axis
    font2 = Dict("color"=>"purple")
    ylabel("Kurtosis",fontdict=font2)
    p = plot_date(L[2:end,:]-1e-9,kur1'[2:end],color="purple",linestyle="-",marker="o",label="Second") # Plot a basic line
    ax2[:set_position](new_position) # Position Method 2
    setp(ax2[:get_yticklabels](),color="purple") # Y Axis font formatting

    fig2[:canvas][:draw]() # Update the figure
    #gcf() # Needed for IJulia to plot inline
    #savefig("multi-axis.")
    
    figure("N_l", (5,4))
    for i=1:length(ls)
        semilogy(ls[i], Nls[i]', "--*")
    end
    xlabel("Level L")
    ylabel("N_l")
    xlim((2,length(L)-1))
    grid(true)
    title("Nr of Samples per Level")
    #legend(["$E[i]" for i=1:length(E)])

    #set(0,'DefaultAxesLineStyleOrder','-*|--*')

    figure("E", (5,4))
    loglog(E,(E.^2).*std_cost)
    loglog(E,(E.^2).*mlmc_cost)
    legend(["Std_MC", "MLMC"])
    title("Cost")
    xlabel("Accuracy eps")
    ylabel("eps^2 Cost")
    grid(true)
    #axis([0.005, 0.1, 50, 1e4])
    #legend(loc="best")
end