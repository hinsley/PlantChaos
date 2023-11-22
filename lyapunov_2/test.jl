using CUDA, StaticArrays


function kernel(a,b,c)
    xix = threadIdx().x + (blockIdx().x - Int32(1)) * blockDim().x
    yix = threadIdx().y + (blockIdx().y - Int32(1)) * blockDim().y
    if xix > length(a) return end
    if yix > length(b) return end
    c[xix,yix] = a[xix] + b[yix]
    return
end
a = cu(collect(1:10))
b = cu(collect(10:10:100))
c = CUDA.zeros(10,10) 

@cuda threads=(10,10) blocks=(1,1) kernel(a,b,c)
