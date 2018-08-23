fixprob(p::Float, eps_val::Float = eps(Float)) = min(max(p, eps_val), Float(1)-eps_val)

fixnan(v::Float) = isnan(v) ? Float(0) : v

fixdiv(x::Float) = x == 0 ? Float(1) : x

logprob(p::Float, eps_val::Float = eps(Float)) = log(fixprob(p, eps_val))

tau(min_samps::Int, min_weight::Float) = (Float(1) - exp(log(Float(1)-min_weight)/min_samps))

function logsumexp{N}(arr::Array{Float, N})  
  add_val = maximum(arr, 1)
  out = log.(sum(exp.(arr .- add_val), 1)) .+ add_val
  return out
end

function probmat{N}(arr::Array{Float, N})  
  p = exp.(arr)
  p ./= sum(p,1)
  return p
end

function set!(rr::RemoteChannel, val)
  if isready(rr)
    take!(rr)
  end
  put!(rr, val)
end

function fetchcall(rr_func::RemoteChannel, args...)
  return fetch(rr_func)(args...)  
end
