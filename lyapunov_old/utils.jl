function _buffered_qr(B::SMatrix, Y) # Y are the deviations
    Q, R = LinearAlgebra.qr(Y)
    return Q, R
end
function _buffered_qr(B::Matrix, Y) # Y are the deviations
    B .= Y
    Q, R = LinearAlgebra.qr!(B)
    return Q, R
end
function set_Q_as_deviations!(tands::TangentDynamicalSystem{true}, Q)
    devs = current_deviations(tands) # it is a view
    if size(Q) ≠ size(devs)
        copyto!(devs, LinearAlgebra.I)
        LinearAlgebra.lmul!(Q, devs)
        set_deviations!(tands, devs)
    else
        set_deviations!(tands, Q)
    end
end

function set_Q_as_deviations!(tands::TangentDynamicalSystem{false}, Q)
    # here `devs` is a static vector
    devs = current_deviations(tands)
    ks = axes(devs, 2) # it is a `StaticArrays.SOneTo(k)`
    set_deviations!(tands, Q[:, ks])
end