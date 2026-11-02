CUDA util
=========

This repository contains a header that defines some useful macros that makes CUDA easier to read and write. 

Exporting
---------
To export a function so that it is accessible from the host through a C ABI, you can write
```cpp
EXPORT void do_something() {
    // do something
}
```
instead of
```cpp
extern "C" __attribute__((visibility("default"))) void do_something() {
    // do something
}
```

For loop
--------
For grid stride loop, you can write
```cpp
GRID_STRIDE(i, n) {
    // do something
}
```
instead of
```cpp
for (int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; 
    i < (n); 
    i += (int64_t)gridDim.x * blockDim.x)
{
    // do something
}
```

For block stride loop, you can write 
```cpp
BLOCK_STRIDE(i, n) {
    // do something
}
```
instead of
```cpp
for (int64_t i = THREAD_RANK; i < (n); i += BLOCK_SIZE)
{
    // do something
}
```

For warp stride loop, you can write 
```cpp
WARP_STRIDE(i, n) {
    // do something
}
```
instead of
```cpp
for (int64_t i = THREAD_RANK / 32; i < (n); i += BLOCK_SIZE / 32)
{
    // do something
}
```

For lane stride loop, you can write 
```cpp
LANE_STRIDE(i, n) {
    // do something
}
```
instead of
```cpp
for (int64_t i = THREAD_RANK % 32; i < (n); i += 32)
{
    // do something
}

