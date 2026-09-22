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
Elapsed: 24.0 s
Speed: 907243384.10 Key/s
Rng Cnt: 3
PrivateKey: 44d83f9967ffffffff
Rmd160: a3f9b683b1c0f30f0b07a9653723aa0833c4d076
================================================
```

> Performance depends on GPU hardware and system configuration.

## Donations

If you find this project useful and would like to support its development, Bitcoin donations are greatly appreciated.

**Bitcoin (BTC)**

```text
bc1p66n8959w9hfkvp77upe5nz846vxvrrcwsmd6plmvn3xy4meuqcgqvc0kyf
```

Thank you for your support!

## License

MIT License
