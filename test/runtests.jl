using Combo
@static if VERSION < v"0.7.0-DEV.2005"
    using Base.Test
else
    using Test
end

# write your own tests here
srand(1234)
a = rand(1:2, 10)
seq, val = cakewalk(seq -> sum((seq - a).^2), 2, length(a), verbose = false);
@test val[1] == Combo.ZERO

