"""
# Black-box combinatorial optimization with the Cakewalk method.

## Usage
`cakewalk(objective_function::Function, M::Int, N::Int, args...)`\n
objective_function - a julia function that returns a value that represents 
the objective value. This function is called by cakewalk in the following fashion:\n 
`objective_value = objective_function(seq, args...)`\n
where `seq::Vector{Int}` is a sequence of integers representing some input.\n

M - each dimension can be a value between 1 and M (for example, M = 2 for binary strings).\n
N - number of dimensions.\n
args - any additional information that might be needed by objective_function.\n

## Parallelism

First, add workers by calling addprocs (for example, `addprocs(number_of_workers)`).\n
Then, make Combo available to all processes: `using Combo`.\n
Lastly, define an objective that is available to all workers. An easy way to achieve 
this is by using the `@everywhere` macro.\n\n
Any additional data that is passed to cakewalk will be passed automatically to the workers by Combo.\n

## Additional important keyword parameters

minimize::Bool - if true minimize, otherwise maximize. Default: true.\n
parallel::Bool - if true, use all worker processes. Default: true.\n

### Convergence

Convergence is determined by comparing two exponentially running averages, or by 
prespefying a number of samples.\n

tot_samps::Int - total number of samples. Default: 0, and is ignored.\n
min_samps::Int - minimal number of samples. \n
max_samps::Int - maximal number of samples. \n

tau_short::Float64 - time constant for the short average. \n
tau_long::Float64 - time constant for the long average. \n
tau_ratio::Float64 - ratio between the former two. Default: 2. If convergence occurs 
too soon increase this value. \n
epsi::Float64 - minimal difference ratio between the two averages. When the 
                     difference ratio is lower than this value, cakewalk stops. 

### Verbosity
verbose::Bool - print last evaluation. Default: true. \n
verbose_fid - where to print. Default: STDOUT. \n
verbose_mod::Int - perform a print whenever number of samples mod verbose_mod is 0. Default: 1. \n 
"""
function cakewalk(objective_function::Function, M::Int, N::Int, args...;
  minimize::Bool = true,

  dist::Dist = Dist(),
  comp_num::Int = max(compnum(dist), 1),

  exhaustive::Bool = false,

  tot_samps::Int = 0,
  min_samps::Int = (tot_samps == 0) ? max(M*N*comp_num, 1000) : tot_samps,
  max_samps::Int = (tot_samps == 0) ? round(Int, 1e6)*M*N : tot_samps,  
  min_weight::Float = Float(.99),  
  tau_short::Float = tau(min_samps,min_weight),
  tau_ratio::Int = 2,
  tau_long::Float = tau(min_samps*tau_ratio,min_weight),
  epsi::Float = Float(1e-2),

  opt_obj::Float = Float(Inf),

  transform_seq::Bool = false, 

  max_init::Float = (comp_num == 1) ? Float(0) : Float(3),

  verbose::Bool = true,
  verbose_fid::Union{Base.TTY, IOStream, Base.PipeEndpoint} = STDOUT,
  verbose_mod::Int = Int(1),
  init_file::String = "",  
  save_dir::String = "",
  save_mod::Int = Int(1e3),

  parallel::Bool = true,

  throw_exceptions::Bool = true,

  kwargs...)
  
  # initializations
  if isempty(init_file)
    opt_seq = zeros(Int, N)    
    opt_val = (Float(Inf), Float(Inf))

    if compnum(dist) != comp_num
      init!(dist, M, N, comp_num, max_init = max_init)
    end

    avg_long = 0. 
    weight_long = 0.

    avg_short = 0. 
    weight_short = 0.

    seq_iter = product([1:M for i in 1:N]...)
    seq_state = start(seq_iter)

    cakewalk_log = []        
  else
    opt_seq, opt_val,
    dist, avg_short, weight_short, avg_long, weight_long, 
    seq_iter, seq_state,
    cakewalk_log = 
    open(init_file, "r") do fid
      deserialize(fid)
    end
  end
  # for the shared array bug when using init_file
  dist = deepcopy(dist)

  function save_data(file_name::String)
    try
      open(file_name, "w") do fid
        serialize(fid,                     
            (opt_seq, opt_val,
            dist, avg_short, weight_short, avg_long, weight_long, 
            seq_iter, seq_state,
            cakewalk_log)
        )
      end
    catch
      println("error saving $file_name")
    end
  end

  # looping  
  loop = true
  rr_loop = RemoteChannel()
  set!(rr_loop, true)

  last_accepted = length(cakewalk_log)
  loop_cond = Condition()
  input_lock = ReentrantLock()  
  result_lock = ReentrantLock()
  exception_lock = ReentrantLock()
  total_samps = length(cakewalk_log)
  has_savedir = !isempty(save_dir) 

  worker_tasks = Tuple{Int, RemoteChannel}[]

  dist_obj = -log(eps(Float))
  diff_ratio = Float(1)

  is_better(a, b, ::Val{true}) = a < b
  is_better(a, b, ::Val{false}) = a > b
  min_type = Val{minimize}()
  min_sign = Float(-1)^minimize
  opt_obj *= min_sign  

  exception = Nullable{Exception}()

  local seq, opt_samp

  function get_input()
    try
      lock(input_lock)   

      if exhaustive
        seq_tuple, seq_state = next(seq_iter, seq_state)
        seq = Int[seq_tuple...]
        comp = -1
      else
        seq, comp = randseq(dist)
      end

      return (seq, comp)

    catch e
      if throw_exceptions
        throw(e)
      end
    finally
      unlock(input_lock)
    end
  end

  function send_result(res_seq, res_seq_tag, res_comp, res_val)
    try
      lock(result_lock)     

      if loop 
        total_samps += 1

        if is_better(res_val[2], opt_val[2], min_type) || ((res_val[2] == opt_val[2]) && is_better(res_val[1], opt_val[1], min_type)) 
          opt_seq = copy(res_seq_tag)
          opt_val = res_val
          opt_samp = total_samps
        end

        if !exhaustive
          dist_obj = update!(dist, transform_seq ? res_seq_tag : res_seq, res_comp, min_sign*Float(res_val[1]); kwargs...)
          
          avg_stat = res_val[1]

          avg_short = (1 - tau_short)*avg_short + tau_short*avg_stat
          weight_short = (1 - tau_short)*weight_short + tau_short  
          avg_short_bar = avg_short/weight_short

          avg_long = (1 - tau_long)*avg_long + tau_long*avg_stat
          weight_long = (1 - tau_long)*weight_long + tau_long  
          avg_long_bar = avg_long/weight_long

          diff_ratio = abs(fixnan((avg_long_bar - avg_short_bar)/avg_long_bar))
        end
        
        file_name = String(has_savedir && (mod(total_samps, save_mod) == 0) ? joinpath(save_dir, string("cakewalk_data_", randstring(), ".jls")) : "")

        if !isempty(file_name)
          save_data(file_name)
        end
        
        if verbose && (mod(total_samps, verbose_mod) == 0)                  
          @printf(verbose_fid, "%-14d %-14.6f %-14.6f %-14.6f %-14.6f %-14.6f %-14.6f %-s\n", total_samps, res_val[1], opt_val[1], opt_val[2], dist_obj, min(weight_short, weight_long), diff_ratio, file_name);
        end

        if !exhaustive
          if (weight_short > min_weight) && (weight_long > min_weight)
            loop = diff_ratio > epsi
          end
          if total_samps >= max_samps
            loop = false
          end
        else
          loop = total_samps < length(seq_iter)
        end

        if !loop
          set!(rr_loop, false)                    
          println()
          notify(loop_cond)
        end

      end

      return :ok
    catch e
      if throw_exceptions
        throw(e)
      end                
    finally
      unlock(result_lock)
    end  
  end

  function send_exception(res_exception) 
    try
      lock(exception_lock)   
      if throw_exceptions 
        loop = false                  
        exception = Nullable{Exception}(res_exception)
        set!(rr_loop, false)                    
        println()
        notify(loop_cond)
      end
    finally
      unlock(exception_lock)
    end
  end             
  # function references
  rr_get_input = RemoteChannel()
  rr_send_result = RemoteChannel()    
  rr_send_exception = RemoteChannel()
  put!(rr_get_input, get_input)
  put!(rr_send_result, send_result)
  put!(rr_send_exception, send_exception)

  # start execution
  if verbose
    @printf(verbose_fid, "%-14s %-14s %-14s %-14s %-14s %-14s %-14s %-14s\n", "sample num", "objective", "opt obj", "opt true obj", "dist obj", "weight", "diff ratio", "file");
  end  

  workers_set = parallel ? workers() : Int[myid()]

  try
    @sync for w in workers_set
      try
        @async rr_worker_task = remotecall_fetch(cakewalk_worker, w, myid(), 
          throw_exceptions,
          rr_loop, rr_get_input, rr_send_result, rr_send_exception,
          objective_function, args...)
        push!(worker_tasks, (w, rr_worker_task))
      catch
      end
    end  

    wait(loop_cond)

    if !isnull(exception)
      throw(get(exception))
    end

  catch e
    if throw_exceptions
      throw(e)
    end
  finally

    # exit workers
    @sync for (w, rr_worker_task) in worker_tasks
      if w in workers_set
        @async remotecall((rr) -> (fetch(rr).exception = InterruptException(); nothing), w, rr_worker_task)          
      end
    end
  end

  if !isempty(save_dir)
    file_name = String(joinpath(save_dir, string("cakewalk_result_", randstring(), ".jls")));
    save_data(file_name)
    @printf("\nSaved - %s\n", file_name);
  end

  return (opt_seq, opt_val)
end

function cakewalk_worker(
    manager_id::Int,
    throw_exceptions::Bool,
    rr_loop::RemoteChannel,
    rr_get_input::RemoteChannel,
    rr_send_result::RemoteChannel,
    rr_send_exception::RemoteChannel,
    objective_function::Function, 
    args...)


  exception = Nullable{Exception}()
  task = 
  @schedule begin
    loop = true
    while loop && fetch(rr_loop)
      try          

        # get input data 
        seq, comp = remotecall_fetch(fetchcall, manager_id, rr_get_input)    

        # perform evalution
        call_result = objective_function(seq, args...)

        # for functions that mutate the sequence
        if isa(call_result, Tuple) && isa(call_result[1], Vector{Int})
          seq_tag = call_result[1]
          if length(call_result) == 3
            call_val = call_result[2:3]
          else
            call_val = call_result[2]
          end
        else
          seq_tag = seq
          call_val = call_result
        end

        # always return a tuple of two values
        val = isa(call_val, Tuple) ? call_val : (call_val, call_val)

        # send back results
        remotecall_fetch(fetchcall, manager_id, rr_send_result, seq, seq_tag, comp, val)              
        
      catch exception
        is_interrupt = isa(exception, InterruptException)
        loop = !throw_exceptions && !is_interrupt
        if !is_interrupt
          remotecall_fetch(fetchcall, manager_id, rr_send_exception, exception)              
        end
      end
    end
  end

  rr_task = RemoteChannel()
  put!(rr_task, task)

  return rr_task
end
