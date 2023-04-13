# Equivariant posteriors

## Nix
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