struct Dist
  item_weights::Vector{Matrix{Float}}  
  aux_item_weights1::Vector{Matrix{Float}}
  aux_item_weights2::Vector{Matrix{Float}}  

  comp_weights::Vector{Float}
  aux_comp_weights1::Vector{Float}  
  aux_comp_weights2::Vector{Float}    

  aux_weight::Vector{Float}  

  sorted_targets::Vector{Float}
  targets::Vector{Float}  
end

const ZERO = Float(0)
const ONE = Float(1)
const TWO = Float(2)

function Dist()
  item_weights = Vector{Matrix{Float}}()
  aux_item_weights1 = Vector{Matrix{Float}}()
  aux_item_weights2 = Vector{Matrix{Float}}()  

  comp_weights = Vector{Float}()  
  aux_comp_weights1 = Vector{Float}()
  aux_comp_weights2 = Vector{Float}()  

  sorted_targets = Vector{Float}()
  targets = Vector{Float}()

  Dist(item_weights, aux_item_weights1, aux_item_weights2, comp_weights, aux_comp_weights1, aux_comp_weights2, Float[0], sorted_targets, targets)
end

function init!(dist::Dist, M::Int, N::Int, compnum::Int; max_init::Float = (compnum == 1) ? ZERO : Float(3))

  if compnum < 1
    error("compnum must be a postive integer")
  end

  resize!(dist.comp_weights, compnum)
  fill!(resize!(dist.comp_weights, compnum), Float(0))
  fill!(resize!(dist.aux_comp_weights1, compnum), Float(0))
  fill!(resize!(dist.aux_comp_weights2, compnum), Float(0))

  resize!(dist.item_weights, compnum)  
  resize!(dist.aux_item_weights1, compnum)    
  resize!(dist.aux_item_weights2, compnum)      

  for i = 1:compnum  
    dist.item_weights[i] = (rand(Float, M, N) - Float(.5))*max_init
    dist.aux_item_weights1[i] = fill(Float(0), M, N)
    dist.aux_item_weights2[i] = fill(Float(0), M, N)    
  end

  dist.aux_weight[1] = Float(0)

  resize!(dist.sorted_targets, 0)
  resize!(dist.targets, 0)

  nothing
end

compnum(dist) = length(dist.comp_weights)

function randcomp(dist::Dist)
  if length(dist.comp_weights) == 0
    error("dist isn't initialized")
  end

  return wsample(1:length(dist.comp_weights), exp.(dist.comp_weights))
end

function randseq(dist::Dist, comp::Int = randcomp(dist))
  if length(dist.item_weights) == 0
    error("dist isn't initialized")
  end

  item_weights = exp.(dist.item_weights[comp])

  M, N = size(item_weights)
  seq = zeros(Int, N)
  for i = 1:N
    seq[i] = wsample(1:M, item_weights[:, i])
  end

  return seq, comp
end

function mapseq(dist::Dist, comp::Int)
  if length(dist.item_weights) == 0
    error("dist isn't initialized")
  end

  item_weights = dist.item_weights[comp]

  M, N = size(item_weights)
  seq = zeros(Int, N)
  for i = 1:N
    seq[i] = indmax(item_weights[:, i])
  end

  return seq
end

function weights(dist::Dist)
  vcat(dist.comp_weights, map(vec, dist.item_weights)...)
end

compprobs(dist::Dist) = probmat(dist.comp_weights)

function itemprobs(dist::Dist)
  return cat(3, map(probmat, dist.item_weights)...)
end

probs(dist::Dist) = vcat(compprobs(dist), vec(itemprobs(dist)))

@enum UpdateType sgd=1 momentum=2 adagrad=3 rmsprop=4 adam=5

function update!(dist::Dist, seq::Vector{Int}, comp::Int, y::Float;   
  targets_num::Int = 100,
  le::Bool = false,
  incd_last::Bool = false,
  early_updates::Bool = false,  
  update::UpdateType = adagrad,
  eta::Float = Float(1e-2),
  eta_items::Float = eta, 
  eta_comps::Float = eta,   
  lambda::Float = Float(0),
  lambda_items::Float = lambda,  
  lambda_comps::Float = lambda,  
  tau1::Float = Float(.9),
  tau2::Float = Float(1e-3),  
  delta::Float = Float(1e-6),
  max_val::Float = Float(88),
  min_val::Float = log(eps(Float)))
  
  # sizes
  M, N = size(dist.item_weights[1])
  C = length(dist.comp_weights)
  n = length(dist.sorted_targets)+1

  if targets_num < 10
    error("targets_num must be higher or equal to 10")
  end

  push!(dist.targets, y)
  update_params = (n > targets_num) || early_updates
  if n > targets_num
    del_y = splice!(dist.targets, 1)
    del_sorted_idx = searchsortedfirst(dist.sorted_targets, del_y)
    splice!(dist.sorted_targets, del_sorted_idx )
    n -= 1
  end

  if le
    idx = searchsortedlast(dist.sorted_targets, y)+1
  else
    idx = searchsortedfirst(dist.sorted_targets, y)
  end
  insert!(dist.sorted_targets, idx, y)
  if n > targets_num
    splice!(dist.sorted_targets, 1)
    idx -= 1
    n -= 1
  end
  # empirical CDF
  py = (n > 1) ? Float((idx - incd_last) / (n - incd_last)) : Float(.5)
  # example weight, distributed as uniform discrete between -1 and 1
  wy = TWO*py - ONE

  # likelihood P(X;w)
  x = falses(M, N)
  for i = 1:N
    x[seq[i], i] = true
  end
  pc = probmat(dist.comp_weights)
  pitems = map(probmat, dist.item_weights)

  # probabilites, log-probalities, and objective
  log_pc = (dist.comp_weights .- logsumexp(dist.comp_weights))
  log_pxgc = map(w -> sum(w[x] - vec(logsumexp(w))), dist.item_weights)
  log_qc = log_pc + log_pxgc
  log_px = logsumexp(log_qc)[1]
  # posterior of C given X
  qc = exp.(log_qc - log_px)
  # correct numerical inaccuracies
  qc ./= sum(qc)

  # for learning the distribution we minimize the following loss
  loss = -wy*log_px

  if update_params
    # update dist weight
    if (update == adagrad) || (update == adam)
      dist.aux_weight[1] += ONE
    elseif (update ==rmsprop) 
      dist.aux_weight[1] = (1 - tau2)*dist.aux_weight[1] + tau2
    end

    # update w[c]
    comp_grad = -wy.*(qc - pc)
    update_field!(dist, comp_grad, :comp_weights, :aux_comp_weights1, :aux_comp_weights2, 
                  Val{update}(), eta_comps, lambda_comps, tau1, tau2, delta, max_val, min_val)

    # update w[i,j,c]
    item_grad = [-wy.*qc[i].*(x - pitems[i]) for i in 1:C]  
    update_field!(dist, item_grad, :item_weights, :aux_item_weights1, :aux_item_weights2,
                  Val{update}(), eta_items, lambda_items, tau1, tau2, delta, max_val, min_val)
  end

  return loss
end

function update_field!(dist::Dist, 
  grad::Array, 
  field_name::Symbol, 
  aux_field_name1::Symbol,
  aux_field_name2::Symbol,  
  ::Val{sgd},
  eta::Float, 
  lambda::Float,  
  tau1::Float, 
  tau2::Float,   
  delta::Float,
  max_val::Float,
  min_val::Float)

  
  for i in 1:length(getfield(dist, field_name))
    getfield(dist, field_name)[i] = min.(max.(
                          (ONE - eta*lambda).*getfield(dist, field_name)[i] - eta.*grad[i]
                          , min_val), max_val)
  end

  return nothing
end

function update_field!(dist::Dist, 
  grad::Array, 
  field_name::Symbol, 
  aux_field_name1::Symbol,
  aux_field_name2::Symbol,  
  ::Val{momentum},
  eta::Float, 
  lambda::Float,  
  tau1::Float, 
  tau2::Float,   
  delta::Float,
  max_val::Float,
  min_val::Float)

  
  for i in 1:length(getfield(dist, field_name))
    getfield(dist, aux_field_name1)[i] = tau1.*getfield(dist, aux_field_name1)[i] + eta.*grad[i]
    getfield(dist, field_name)[i] = min.(max.(        
                          (ONE - eta*lambda).*getfield(dist, field_name)[i] - getfield(dist, aux_field_name1)[i]
                          , min_val), max_val)      
  end

  return nothing
end

function update_field!(dist::Dist, 
  grad::Array, 
  field_name::Symbol, 
  aux_field_name1::Symbol,
  aux_field_name2::Symbol,  
  ::Val{adagrad},
  eta::Float, 
  lambda::Float,  
  tau1::Float, 
  tau2::Float,   
  delta::Float,
  max_val::Float,
  min_val::Float)

  
  for i in 1:length(getfield(dist, field_name))
    getfield(dist, aux_field_name1)[i] .+= grad[i].^2  
    aux = sqrt.(getfield(dist, aux_field_name1)[i]./dist.aux_weight[1]) .+ delta        

    getfield(dist, field_name)[i] = min.(max.(
                          (aux.*getfield(dist, field_name)[i] - eta.*grad[i])./(aux .+ eta*lambda)
                          , min_val), max_val)
  end

  return nothing
end

function update_field!(dist::Dist, 
  grad::Array, 
  field_name::Symbol, 
  aux_field_name1::Symbol,
  aux_field_name2::Symbol,  
  ::Val{rmsprop},
  eta::Float, 
  lambda::Float,  
  tau1::Float, 
  tau2::Float,   
  delta::Float,
  max_val::Float,
  min_val::Float)
  
  for i in 1:length(getfield(dist, field_name))
    getfield(dist, aux_field_name1)[i] = (1 - tau2).*getfield(dist, aux_field_name1)[i] + tau2.*grad[i].^2  
    aux = sqrt.(getfield(dist, aux_field_name1)[i]./dist.aux_weight[1]) .+ delta

    getfield(dist, field_name)[i] = min.(max.(
                          (aux.*getfield(dist, field_name)[i] - eta.*grad[i])./(aux .+ eta*lambda)
                          , min_val), max_val)
  end

  return nothing
end

function update_field!(dist::Dist, 
  grad::Array, 
  field_name::Symbol, 
  aux_field_name1::Symbol,
  aux_field_name2::Symbol,  
  ::Val{adam},
  eta::Float, 
  lambda::Float,  
  tau1::Float, 
  tau2::Float,   
  delta::Float,
  max_val::Float,
  min_val::Float)
  
  for i in 1:length(getfield(dist, field_name))
    est_fix1 = ONE/(ONE - (ONE - tau1)^dist.aux_weight[1])
    est_fix2 = ONE/(ONE - (ONE - tau2)^dist.aux_weight[1])

    getfield(dist, aux_field_name1)[i] = (ONE - tau1).*getfield(dist, aux_field_name1)[i] + tau1.*grad[i]
    getfield(dist, aux_field_name2)[i] = (ONE - tau2).*getfield(dist, aux_field_name2)[i] + tau2.*grad[i].^2  

    aux = sqrt.(max.(est_fix2*getfield(dist, aux_field_name2)[i],0)) .+ delta  
    getfield(dist, field_name)[i] = min.(max.(
                                        (aux.*getfield(dist, field_name)[i] - (eta*est_fix1).*getfield(dist, aux_field_name1)[i])./(aux .+ eta*lambda)
                                        , min_val), max_val)      
  end

  return nothing
end
