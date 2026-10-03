// RocketKV research kernels. See docs/rocketkv/NOTICE.
#include "common.h"

static float rocket_read(device const char * data, ulong offset, bool f16) {
    return f16 ? float(*(device const half *)(data + offset)) : *(device const float *)(data + offset);
}

kernel void kernel_rocketkv_reduce(
        constant ggml_metal_kargs_rocketkv & a [[buffer(0)]],
        device const char * src [[buffer(1)]],
        device float * dst [[buffer(6)]],
        uint i [[thread_position_in_grid]]) {
    if (i >= uint(a.n0*a.n1)) {
        return;
    }
    const int t = i % a.n0, h = i/a.n0, group = a.a2/a.n1;
    float sum = 0;
    for (int g = 0; g < group; ++g) {
        for (int w = 0; w < a.a1; ++w) {
            sum += *(device const float *)(src + t*a.ab0 + w*a.ab1 + (h*group + g)*a.ab2);
        }
    }
    dst[i] = sum;
}

static bool rocket_before(device const char * src, int x, int y, int n, ulong stride) {
    if (x >= n || y >= n) {
        return x < y;
    }
    const float vx = *(device const float *)(src + x*stride);
    const float vy = *(device const float *)(src + y*stride);
    return vx > vy || (vx == vy && x < y);
}

kernel void kernel_rocketkv_top_k(
        constant ggml_metal_kargs_rocketkv & a [[buffer(0)]],
        device const char * src [[buffer(1)]],
        device int * dst [[buffer(6)]],
        threadgroup int * ids [[threadgroup(0)]],
        uint h [[threadgroup_position_in_grid]],
        uint tid [[thread_position_in_threadgroup]],
        uint nt [[threads_per_threadgroup]]) {
    int padded = 1;
    while (padded < a.a0) {
        padded *= 2;
    }
    src += h*a.ab1;
    for (uint i = tid; i < uint(padded); i += nt) {
        ids[i] = i;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int width = 2; width <= padded; width *= 2) {
        for (int step = width/2; step > 0; step /= 2) {
            for (uint i = tid; i < uint(padded); i += nt) {
                const uint j = i ^ step;
                if (j > i) {
                    const bool swap = (i & width) == 0 ?
                        rocket_before(src, ids[j], ids[i], a.a0, a.ab0) :
                        rocket_before(src, ids[i], ids[j], a.a0, a.ab0);
                    if (swap) {
                        const int x = ids[i];
                        ids[i] = ids[j];
                        ids[j] = x;
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
    for (uint i = tid; i < uint(a.n0); i += nt) {
        dst[h*a.n0 + i] = ids[i];
    }
}

kernel void kernel_rocketkv_indices(
        constant ggml_metal_kargs_rocketkv & a [[buffer(0)]],
        device const int * src [[buffer(1)]],
        device int * dst [[buffer(6)]],
        uint2 i [[thread_position_in_grid]]) {
    if (i.x >= uint(a.n0)) {
        return;
    }
    if (i.x >= uint(a.a0)) {
        dst[i.y*a.n0 + i.x] = a.p2 - a.p1 + i.x - a.a0;
        return;
    }
    const int x = src[i.y*a.a0 + i.x];
    int rank = 0;
    for (int j = 0; j < a.a0; ++j) {
        rank += src[i.y*a.a0 + j] < x;
    }
    dst[i.y*a.n0 + rank] = x;
}

kernel void kernel_rocketkv_metadata(
        constant ggml_metal_kargs_rocketkv & a [[buffer(0)]],
        device const char * keys [[buffer(1)]],
        device const int * live [[buffer(2)]],
        device char * dst [[buffer(6)]],
        uint i [[thread_position_in_grid]]) {
    const int pages = a.n0, dim = a.a0, n = live[0], page = a.p1;
    if (i >= uint(dim*a.a1*(a.p2 ? pages : 1))) {
        return;
    }
    const int p = a.p2 ? i % pages : (n - 1)/page;
    const int d = a.p2 ? (i/pages) % dim : i % dim;
    const int h = a.p2 ? i/(pages*dim) : i/dim;
    const int imin = p + pages*(d + 2*dim*h), imax = imin + pages*dim;
    float lo = INFINITY, hi = -INFINITY;
    int first = p*page;
    if (!a.p2 && (n - 1) % page != 0) {
        lo = rocket_read(dst, imin*(a.f16 ? 2 : 4), a.f16);
        hi = rocket_read(dst, imax*(a.f16 ? 2 : 4), a.f16);
        first = n - 1;
    }
    for (int t = first; t < min(n, (p + 1)*page); ++t) {
        const float x = rocket_read(keys, d*a.ab0 + h*a.ab1 + t*a.ab2, a.f16);
        lo = min(lo, x);
        hi = max(hi, x);
    }
    if (a.f16) {
        ((device half *) dst)[imin] = half(lo);
        ((device half *) dst)[imax] = half(hi);
    } else {
        ((device float *) dst)[imin] = lo;
        ((device float *) dst)[imax] = hi;
    }
}

kernel void kernel_rocketkv_query(
        constant ggml_metal_kargs_rocketkv & a [[buffer(0)]],
        device const char * query [[buffer(1)]],
        device float * dst [[buffer(6)]],
        uint i [[thread_position_in_grid]]) {
    if (i >= uint(a.n0*a.n2)) {
        return;
    }
    const int d = i % a.n0, h = i/a.n0, group = a.a1/a.n2;
    float sum = 0, magnitude = 0;
    for (int g = 0; g < group; ++g) {
        const float q = *(device const float *)(query + d*a.ab0 + (h*group + g)*a.ab1);
        sum += q;
        magnitude += abs(q);
    }
    dst[d + a.n0*2*h] = magnitude;
    dst[d + a.n0*(2*h + 1)] = sum;
}

kernel void kernel_rocketkv_scores(
        constant ggml_metal_kargs_rocketkv & a [[buffer(0)]],
        device const char * query [[buffer(1)]],
        device const char * metadata [[buffer(2)]],
        device const int * dims [[buffer(3)]],
        device const float * sums [[buffer(4)]],
        device const int * live [[buffer(5)]],
        device float * dst [[buffer(6)]],
        uint i [[thread_position_in_grid]]) {
    const int pages = a.b0, dim = a.a0, page = a.p1;
    if (i >= uint(pages*a.a1)) {
        return;
    }
    const int p = i % pages, h = i/pages, kh = h/(a.a1/a.b2), n = live[0];
    float dot = 0, selected = 0, total = 0;
    if (p*page < n) {
        for (int d = 0; d < dim; ++d) {
            total += abs(*(device const float *)(query + d*a.ab0 + h*a.ab1));
        }
        for (int r = 0; r < a.c0; ++r) {
            const int d = dims[kh*a.c0 + r];
            const float q = *(device const float *)(query + d*a.ab0 + h*a.ab1);
            const int signed_d = d + (sums[d + dim*(2*kh + 1)] > 0 ? dim : 0);
            const int index = p + pages*(signed_d + 2*dim*kh);
            const float key = rocket_read(metadata, index*(a.f16 ? 2 : 4), a.f16);
            dot += q*key;
            selected += abs(q);
        }
    }
    const float scale = selected > 0 ? sqrt(dim*selected/total) : 1.0f;
    for (int t = p*page; t < min(a.n0, (p + 1)*page); ++t) {
        dst[h*a.n0 + t] = t < n ? dot/scale : -INFINITY;
    }
}
