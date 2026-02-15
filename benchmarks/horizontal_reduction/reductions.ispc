// ISPC horizontal reduction kernels

export uniform float sum_ispc(
    uniform float data[],
    uniform int32 len
) {
    float partial = 0.0f;
    foreach (i = 0 ... len) {
        partial += data[i];
    }
    return reduce_add(partial);
}

export uniform float max_ispc(
    uniform float data[],
    uniform int32 len
) {
    float partial = data[0];
    foreach (i = 0 ... len) {
        partial = max(partial, data[i]);
    }
    return reduce_max(partial);
}

export uniform float min_ispc(
    uniform float data[],
    uniform int32 len
) {
    float partial = data[0];
    foreach (i = 0 ... len) {
        partial = min(partial, data[i]);
    }
    return reduce_min(partial);
}
