#include <cmath>
#include <span>

namespace {

/// Based on https://arxiv.org/abs/math/0505419
template <typename T>
T half_width_mode_impl(std::span<const T> x) {
  switch (x.size()) {
    case 1: {
      return x.front();
    }
    case 2: {
      return 0.5 * (x.front() + x.back());
    }
    case 3: {
      if (x[2] - x[1] < x[1] - x[0]) {
        return 0.5 * (x[2] + x[1]);
      } else if (x[2] - x[1] > x[1] - x[0]) {
        return 0.5 * (x[1] + x[0]);
      } else {
        return x[1];
      }
    }
    default: {
      T wmin = x.back() - x.front();

      // If all elements are equal
      if (wmin == 0.0) {
        return x.front();
      }

      const std::size_t N = std::ceil(x.size() / 2.0);

      std::size_t j = 0;
      for (auto i = 0ul; i < x.size() - N; ++i) {
        const T w = x[i + N - 1] - x[i];
        if (w < wmin) {
          wmin = w;
          j = i;
        }
      }

      return half_width_mode_impl(
          std::span<const T>(x.begin() + j, x.begin() + j + N - 1));
    }
  }
}

}  // namespace

extern "C" double half_width_mode(const double *x, std::size_t s) {
  return half_width_mode_impl(std::span<const double>(x, s));
}

extern "C" float half_width_mode_float(const float *x, std::size_t s) {
  return half_width_mode_impl(std::span<const float>(x, s));
}
