// RocketKV research implementation. See docs/rocketkv/NOTICE.
#pragma once

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace rocketkv_ref {

struct tensor {
    int h, n, d;
    std::vector<float> data;

    tensor(int heads, int tokens, int dim) : h(heads), n(tokens), d(dim), data(heads*tokens*dim) {}
    float & at(int head, int token, int dim) { return data[(head*n + token)*d + dim]; }
    float at(int head, int token, int dim) const { return data[(head*n + token)*d + dim]; }
};

inline std::vector<float> softmax(std::vector<float> x) {
    const float m = *std::max_element(x.begin(), x.end());
    double sum = 0;
    for (auto & v : x) {
        v = std::exp(v - m);
        sum += v;
    }
    for (auto & v : x) {
        v /= sum;
    }
    return x;
}

inline std::vector<int> topk(const std::vector<float> & scores, int k) {
    if (k < 1 || k > (int) scores.size()) {
        throw std::invalid_argument("invalid top-k");
    }
    std::vector<int> ids(scores.size());
    std::iota(ids.begin(), ids.end(), 0);
    std::partial_sort(ids.begin(), ids.begin() + k, ids.end(), [&](int a, int b) {
        return scores[a] > scores[b] || (scores[a] == scores[b] && a < b);
    });
    ids.resize(k);
    return ids;
}

inline tensor gather(const tensor & src, const std::vector<int> & ids) {
    tensor dst(src.h, ids.size()/src.h, src.d);
    for (int h = 0; h < src.h; ++h) {
        for (int t = 0; t < dst.n; ++t) {
            const int i = ids[h*dst.n + t];
            if (i < 0 || i >= src.n) {
                throw std::invalid_argument("gather index out of range");
            }
            for (int d = 0; d < src.d; ++d) {
                dst.at(h, t, d) = src.at(h, i, d);
            }
        }
    }
    return dst;
}

struct snap_result {
    std::vector<float> scores, pooled;
    std::vector<int> indices;
};

inline snap_result snap(const tensor & q, const tensor & k, int capacity, int kernel) {
    if (q.h % k.h || q.d != k.d || q.n > k.n || capacity < q.n || capacity > k.n || kernel < 1 || kernel % 2 == 0) {
        throw std::invalid_argument("invalid SnapKV shape or parameters");
    }
    snap_result out;
    const int early = k.n - q.n;
    const int group = q.h/k.h;
    out.scores.resize(k.h*early);
    out.pooled.resize(k.h*early);
    for (int h = 0; h < q.h; ++h) {
        for (int w = 0; w < q.n; ++w) {
            std::vector<float> score(early + w + 1);
            for (int t = 0; t < (int) score.size(); ++t) {
                for (int d = 0; d < k.d; ++d) {
                    score[t] += q.at(h, w, d)*k.at(h/group, t, d);
                }
                score[t] /= std::sqrt(float(k.d));
            }
            score = softmax(score);
            for (int t = 0; t < early; ++t) {
                out.scores[(h/group)*early + t] += score[t];
            }
        }
    }
    for (int h = 0; h < k.h; ++h) {
        std::vector<float> pooled(early);
        for (int t = 0; t < early; ++t) {
            float best = -INFINITY;
            for (int i = std::max(0, t - kernel/2); i <= std::min(early - 1, t + kernel/2); ++i) {
                best = std::max(best, out.scores[h*early + i]);
            }
            pooled[t] = out.pooled[h*early + t] = best;
        }
        std::vector<int> ids;
        if (capacity > q.n) {
            ids = topk(pooled, capacity - q.n);
            std::sort(ids.begin(), ids.end());
        }
        for (int t = early; t < k.n; ++t) {
            ids.push_back(t);
        }
        out.indices.insert(out.indices.end(), ids.begin(), ids.end());
    }
    return out;
}

struct metadata {
    tensor min, max;
    int page;

    metadata(int h, int n, int d, int p) : min(h, (n + p - 1)/p, d), max(h, (n + p - 1)/p, d), page(p) {
        std::fill(min.data.begin(), min.data.end(), INFINITY);
        std::fill(max.data.begin(), max.data.end(), -INFINITY);
    }
};

inline void update(metadata & m, const tensor & k, int begin) {
    for (int h = 0; h < k.h; ++h) {
        for (int t = begin; t < k.n; ++t) {
            for (int d = 0; d < k.d; ++d) {
                m.min.at(h, t/m.page, d) = std::min(m.min.at(h, t/m.page, d), k.at(h, t, d));
                m.max.at(h, t/m.page, d) = std::max(m.max.at(h, t/m.page, d), k.at(h, t, d));
            }
        }
    }
}

inline tensor attention(const tensor & q, const tensor & k, const tensor & v) {
    tensor out(q.h, 1, q.d);
    const int group = q.h/k.h;
    for (int h = 0; h < q.h; ++h) {
        std::vector<float> scores(k.n);
        for (int t = 0; t < k.n; ++t) {
            for (int d = 0; d < k.d; ++d) {
                scores[t] += q.at(h, 0, d)*k.at(h/group, t, d);
            }
            scores[t] /= std::sqrt(float(k.d));
        }
        scores = softmax(scores);
        for (int t = 0; t < k.n; ++t) {
            for (int d = 0; d < k.d; ++d) {
                out.at(h, 0, d) += scores[t]*v.at(h/group, t, d);
            }
        }
    }
    return out;
}

inline std::vector<int> exact_topk(const tensor & q, const tensor & k, int count) {
    const int group = q.h/k.h;
    std::vector<int> ids;
    for (int kh = 0; kh < k.h; ++kh) {
        std::vector<float> weights(k.n);
        for (int g = 0; g < group; ++g) {
            std::vector<float> scores(k.n);
            for (int t = 0; t < k.n; ++t) {
                for (int d = 0; d < k.d; ++d) {
                    scores[t] += q.at(kh*group + g, 0, d)*k.at(kh, t, d);
                }
                scores[t] /= std::sqrt(float(k.d));
            }
            scores = softmax(scores);
            for (int t = 0; t < k.n; ++t) {
                weights[t] += scores[t];
            }
        }
        const auto selected = topk(weights, count);
        ids.insert(ids.end(), selected.begin(), selected.end());
    }
    return ids;
}

struct hsa_result {
    std::vector<int> dims, indices;
    std::vector<float> page_scores, token_scores;
    tensor keys, values, output;

    hsa_result(int hkv, int hq, int k, int d) : keys(hkv, k, d), values(hkv, k, d), output(hq, 1, d) {}
};

inline hsa_result hsa(const tensor & q, const tensor & k, const tensor & v, const metadata & m, int r, int count) {
    if (q.n != 1 || q.h % k.h || q.d != k.d || v.data.size() != k.data.size() || r < 1 || r > k.d || count < 1 || count > k.n) {
        throw std::invalid_argument("invalid HSA shape or parameters");
    }
    hsa_result out(k.h, q.h, count, k.d);
    const int group = q.h/k.h;
    out.page_scores.resize(q.h*m.min.n);
    out.token_scores.resize(k.h*k.n);
    for (int kh = 0; kh < k.h; ++kh) {
        std::vector<float> magnitude(k.d), sign(k.d);
        for (int g = 0; g < group; ++g) {
            for (int d = 0; d < k.d; ++d) {
                const float x = q.at(kh*group + g, 0, d);
                magnitude[d] += std::abs(x);
                sign[d] += x;
            }
        }
        const auto dims = topk(magnitude, r);
        out.dims.insert(out.dims.end(), dims.begin(), dims.end());
        std::vector<float> scores(k.n);
        for (int g = 0; g < group; ++g) {
            const int h = kh*group + g;
            float selected = 0, total = 0;
            for (int d = 0; d < k.d; ++d) {
                total += std::abs(q.at(h, 0, d));
            }
            for (int d : dims) {
                selected += std::abs(q.at(h, 0, d));
            }
            const float scale = selected > 0 ? std::sqrt(k.d*selected/total) : 1.0f;
            std::vector<float> raw(k.n);
            for (int p = 0; p < m.min.n; ++p) {
                float dot = 0;
                for (int d : dims) {
                    const float key = sign[d] > 0 ? m.max.at(kh, p, d) : m.min.at(kh, p, d);
                    dot += q.at(h, 0, d)*key;
                }
                out.page_scores[h*m.min.n + p] = dot/scale;
                for (int t = p*m.page; t < std::min(k.n, (p + 1)*m.page); ++t) {
                    raw[t] = dot/scale;
                }
            }
            raw = softmax(raw);
            for (int t = 0; t < k.n; ++t) {
                scores[t] += raw[t];
            }
        }
        std::copy(scores.begin(), scores.end(), out.token_scores.begin() + kh*k.n);
        const auto ids = topk(scores, count);
        out.indices.insert(out.indices.end(), ids.begin(), ids.end());
    }
    out.keys = gather(k, out.indices);
    out.values = gather(v, out.indices);
    out.output = attention(q, out.keys, out.values);
    return out;
}

} // namespace rocketkv_ref
