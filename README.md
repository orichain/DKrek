# DKrek

High-performance CUDA-based 9-byte Bitcoin Puzzle private key solver.

## Requirements

Recommended environment:

- Rocky Linux 10.2
- NVIDIA Driver installed from:
  https://www.nvidia.com/en-us/drivers/
- CUDA Toolkit installed from:
  https://developer.nvidia.com/cuda-downloads
- GCC
- GNU Make

## Build

```bash
gmake clean all
```

## Run

```bash
./DKrek
```

## Example Output (RTX 2060 SUPER)

```text
==============.....SEARCHING......==============
Rmd160: f6f5431d25bbf7b12e8add9af5e3475c44a0a5b8
==============.....SEARCHING......==============
Elapsed: 21h 10m
Kecepatan: 824709303.79 Keys/s
PrivateKey: 7a295be40affffffff
Rmd160: 0d643b59186a591131c7c1b27ff26b2868b37a53
================================================
```

> Performance depends on GPU hardware and system configuration.

## Donations

If you find this project useful and would like to support its development, Bitcoin donations are greatly appreciated.

**Bitcoin (BTC)**

```text
bc1qzfef9zjmhh44qhg4xtzh04u0vtuvjpc4kg8zalv8mvsf3ppt7krqtlvsx6
```

Thank you for your support!

## License

MIT License
