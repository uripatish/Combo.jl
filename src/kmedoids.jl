###############################
# kmedoids, Voronoi iteration #
###############################

function kmedoids_voronoi(D::Matrix{Float}, k::Int;
                    verbose::Bool = true,
                    epsi::Float = Float(1e-10),
                    avg_iters::Int = Int(1),
                    max_iters::Int = Int(1000),
                    calc_emp::Bool = true,
                    medoid_init::Vector{Int} = Int[],
                    init::Symbol = :per)

  n = size(D,1)

  # initialize optimization
  opt = Float[]
  queue = Array{Float, 1}()
  diff_ratio = Float(1)
  avg_objective = Float(0)
  medoid_idx = zeros(Int, k)

  max_val = maximum(D)
  if verbose
    println("initialization...")
  end
  if init == :rand
    init_idx = 1:n
  elseif init == :cen
    ~, init_idx = findmin(max(D,max_val*eye(n)),2)
    ~, init_idx = ind2sub(size(D), init_idx[:])
  elseif init == :per
    ~, init_idx = findmax(D,2)
    ~, init_idx = ind2sub(size(D), init_idx[:])
  else
    error("no such initialization")
  end
  medoid_idx = (length(medoid_init) == k) ? medoid_init : sample(init_idx, k, replace = false)
  ~,clust_idx = findmin(D[:, medoid_idx],2)

  if verbose
    @printf("%-20s %-20s %-20s\n", "iteration", "objective", "difference ratio");
  end
  local clust_idx
  local objective
  local min_clust_idx, min_medoid_idx, t  
  min_obj = 1.
  mu = mean(D)  
  for t in 0:max_iters

    if t > 0
      # find new medoids
      for c in 1:k
        cidx = find(clust_idx .== c)
        medoid_idx[c] = (length(cidx) > 0) ? cidx[indmin(mean(D[cidx, cidx],1))] : 0
      end
      # reassign empty medoids
      empt_ind = medoid_idx .== 0
      empt_num = sum(empt_ind)
      if empt_num > 0
        medoid_idx[empt_ind] = sample(init_idx, empt_num, replace = (length(init_idx) <= empt_num))
      end
    end

    # nearest cluster
    min_dist,clust_idx = findmin(D[:, medoid_idx],2)
    ~, clust_idx = ind2sub(size(D), clust_idx[:])
    # objective
    objective = mean(fixnan.(min_dist./mu))

    if objective < min_obj
      min_obj = objective
      min_clust_idx = clust_idx
      min_medoid_idx = medoid_idx
    end

    if t > 0
      # finalization
      Base.push!(opt, objective)

      if t > avg_iters
        diff_ratio = abs(objective - avg_objective)/fixdiv(avg_objective);
      end

      Base.push!(queue, objective/avg_iters)
      avg_objective = avg_objective + objective/avg_iters
      if t > avg_iters
        avg_objective = avg_objective - shift!(queue)
      end
    end

    if verbose
      if t > avg_iters
        @printf("%-20d %-20.6f %-20.6f\n", t, objective, diff_ratio)
      else
        @printf("%-20d %-20.6f %-20s\n", t, objective, "-")
      end
    end

    if (diff_ratio < epsi)
      break
    end

  end

  return min_clust_idx, min_medoid_idx, min_obj, t
end

#################
# kmedoids, PAM #
#################

function kmedoids_pam(D::Matrix{Float}, k::Int;
                    verbose::Bool = true,
                    avg_iters::Int = Int(10),
                    max_iters::Int = Int(1000),
                    min_obj::Float = eps(Float),
                    calc_emp::Bool = true,
                    medoid_init::Vector = Int[],
                    init::Symbol = :per)

  n = size(D,1)

  # initialize optimization
  opt = Float[]
  queue = Array{Float, 1}()
  diff_ratio = Float(1)
  avg_objective = Float(0)
  medoid_idx = zeros(Int, k)

  max_val = maximum(D)
  if length(medoid_init) != k
    if verbose
      println("initialization...")
    end
    if init == :rand
      init_idx = 1:n
    elseif init == :cen
      ~, init_idx = findmin(max(D,max_val*eye(n)),2)
      ~, init_idx = ind2sub(size(D), init_idx[:])
    elseif init == :per
      ~, init_idx = findmax(D,2)
      ~, init_idx = ind2sub(size(D), init_idx[:])
    else
      error("no such initialization")
    end
    medoid_idx = sample(init_idx, k, replace = false)
  else
    medoid_idx = deepcopy(medoid_init)
  end
  mu = mean(D)
  local clust_idx
  local objective
  new_min_dist = zeros(n)

  if verbose
    @printf("%-20s %-20s\n", "iteration", "objective");
  end

  total_samp = 0
  best_samp = 0
  switched = true
  t = 0
  while (switched) && (t<=max_iters)
    t += 1

    ###############
    # build phase #
    ###############

    # nearest mediod
    min_dist,clust_idx = findmin(D[:, medoid_idx],2)
    ~, clust_idx = ind2sub(size(D), clust_idx[:])
    # objective
    objective = fixnan(mean(min_dist)./mu)
    total_samp += 1
    best_samp = total_samp

    if verbose
      if t > avg_iters
        @printf("%-20d %-20.6f\n", t, objective)
      else
        @printf("%-20d %-20s\n", t, objective)
      end
    end

    if objective < min_obj
      break
    end

    ##############
    # swap phase #
    ##############

    # find new medoids
    switched = false
    for i in randperm(k)
      new_min_dist[:] = minimum(D[:, medoid_idx[1:k .!= i]],2)
      for j in setdiff(randperm(n), medoid_idx)
        new_objective = fixnan(mean(min.(new_min_dist, D[:, j]))./mu)
        total_samp += 1
        if new_objective < objective
          switched = true
          medoid_idx[i] = j
        end
        if switched
          break
        end
      end
      if switched
        break
      end
    end
  end

  return clust_idx, medoid_idx, objective, best_samp, total_samp
end

nothing
