#pragma once
#include <vector>
#include <cstdlib> // Required for std::aligned_alloc
#include <new>     // Required for std::bad_alloc

// 1. The Custom Allocator
template <typename T>
struct AlignedAllocator {
  using value_type = T;

  AlignedAllocator() noexcept = default;
  template <typename U> AlignedAllocator(const AlignedAllocator<U>&) noexcept {}

  T* allocate(std::size_t n) {
    std::size_t bytes = n * sizeof(T);

    // std::aligned_alloc strictly requires the requested size to be a
    // multiple of the alignment. This bitwise math pads the size up to
    // the nearest multiple of 32.
    std::size_t padded_bytes = (bytes + 31) & ~31;

    // Request 32-byte aligned memory
    void* p = std::aligned_alloc(32, padded_bytes);

    if (!p) {
        throw std::bad_alloc();
    }
    return static_cast<T*>(p);
  }

  void deallocate(T* p, std::size_t) noexcept {
    std::free(p); // Must use standard free() to release aligned_alloc memory
  }
};

// 2. The Matrix Class
class Matrix {
private:
  int row;
  int colum;

  // 3. Inject the custom allocator into the vector
  std::vector<float, AlignedAllocator<float>> data;

public:
  // Matrix Constructor
  Matrix(int row, int colum, float initial_value = 0.0f)
      : row(row), colum(colum),
        data(static_cast<std::size_t>(row * colum), initial_value) {};

  float &operator()(int r, int c) { return data.at((r * colum) + c); };

  const float &operator()(int r, int c) const {
    return data.at((r * colum) + c);
  };

  int get_rows() const;
  int get_colum() const;
  int get_size() const;

  void print_matrix();
  void randomize();

  // 4. Expose the raw data pointer for our future AVX instructions
  float* get_data() { return data.data(); }
  const float* get_data() const { return data.data(); }
};
