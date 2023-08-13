[![CI](https://github.com/hlinander/equivariant-posteriors/actions/workflows/main.yml/badge.svg)](https://github.com/hlinander/equivariant-posteriors/actions/workflows/main.yml)

# Equivariant posteriors
Test bed for equivariant posterior project.

Train and evaluate PyTorch models with reproducibility in mind.

- Computational environment reproducible through Nix flake.
- Python based configuration in terms of dataclasses.
- Convenient metric functionality with focus on saving as much as possible for future inspection.
- Simple TUI for easy progress inspection.
- Postgres storage with experimental Apache Superset couplings for easy visualization and run tracking.

## Nix
This project uses [Nix](https://nixos.org/) to manage development and runtime dependencies.

### Install
Install nix to your home-folder or system wide.
```
  https://nixos.org/download.html#download-nix
```
Enable [flakes](https://zero-to-nix.com/concepts/flakes) by one of

 1. Nix in other distribution: Create (or add to) `~/.config/nix/nix.conf` (or `/etc/nix/nix.conf`)
    ```
      experimental-features = nix-command flakes
    ```
 2. NixOS: Add to `/etc/nixos/configuration.nix`
    ```
      nix = {
        package = pkgs.nixFlakes;
        extraOptions = ''
          experimental-features = nix-command flakes
        '';
      };
    ```
### CUDA binary cache
Add the binary cache for CUDA enabled packages at
https://app.cachix.org/cache/cuda-maintainers#pull

### Development 
Start a development shell
```
  nix develop
```

### Test project
To build and run tests
```
  nix build
```

### Build singularity image
Build a singularity image with CUDA support containing the project
```
  nix build .#sing
```
