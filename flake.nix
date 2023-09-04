{
  # inputs.nixpkgs.url = "github:NixOS/nixpkgs";
  inputs.helix-pkg.url = "github:helix-editor/helix";
  # inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-22.11";
  # inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-23.05";
  outputs = { self, nixpkgs, helix-pkg, ... }:
  # outputs = { self, nixpkgs, ... }:
    let
      disableCudaEnvFlag = builtins.getEnv "DISABLE_CUDA";
      system = "x86_64-linux";
      pkgs = (import nixpkgs {
        inherit system;
        config = {
          allowUnfree = true;
          cudaSupport = disableCudaEnvFlag != "1";
        };
      });
      helixmaster = helix-pkg.packages.${system}.default;
      # myPython3 = pkgs.python3.override {
      #   packageOverrides = pyself: pysuper: {
      #     torchUncertainty = pysuper.buildPythonPackage rec {
      #       pname = "torch_uncertainty";
      #       version = "0.1.4";
      #       src = pysuper.fetchPypi {
      #         inherit pname version;
      #         sha256 = "sha256-4tPBN8ADtBfJ2ffHdf5dcuyXiopv6BM7aLt7fNXLCx0="; # TODO
      #       };
      #       propagatedBuildInputs = [ 
      #        pyself.pytorch-lightning
      #       ];
      #       doCheck = false;
      #     };
      #   };
      # };
      program = pkgs.python3Packages.buildPythonApplication {
        pname = "equivariant-transformers";
        version = "1.0";

        propagatedBuildInputs = pkgs.python3.withPackages(p: [#with pkgs.python3Packages; [
          p.tqdm
          p.numpy
          p.pytorch
          p.torchvision
          p.torchmetrics
          p.plotext
          p.wandb
          p.pandas
          p.psycopg
          p.pytest
        ]);

        src = ./.;

        checkPhase = ''
          # ${pkgs.python3Packages.pytorch}/bin/torchrun --standalone --nnodes=1 --nproc_per_node=1 test.py
          export TORCH_DEVICE="cpu"
          python test.py
        '';
      };
      devinputs = with pkgs; [
          ruff
          helixmaster
          nixfmt
          jq
          (rWrapper.override{ packages = with rPackages; [ ggplot2 dplyr latex2exp patchwork reticulate Hmisc]; })
          #(rstudioWrapper.override{ packages = with rPackages; [ ggplot2 dplyr patchwork reticulate Hmisc]; })
          # cudatoolkit
          (python3.withPackages (p: [
            (p.rpy2.override{ extraRPackages = with rPackages; [ggplot2 dplyr latex2exp patchwork reticulate Hmisc]; })
            # p.torchUncertainty
            p.gitpython
            p.tqdm
            p.python-lsp-server
            p.python-lsp-ruff
            p.numpy
            p.pytorch
            # (p.torchvision.override {torch = p.pytorch-bin;})
            p.torchvision
            p.plotext
            p.torchmetrics
            # (p.torchmetrics.override {torch = p.pytorch-bin;})
            p.ipython
            p.black
            p.flake8
           # p.gitpython
            p.wandb
            p.snakeviz
            p.pandas
            p.matplotlib
            p.plotnine
            p.psycopg
            p.pytest
          ]))
        ];
    in {
      packages.x86_64-linux.default = program;
      packages.aarch64-darwin.default = program;
      devShells.x86_64-linux.default = pkgs.mkShellNoCC {
        buildInputs = devinputs;
        nativeBuildInputs = [pkgs.cudatoolkit];
        shellHook = '' 
          export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/run/opengl-driver/lib/"
          # export CUDA_PATH=${pkgs.cudatoolkit}
        '';
         };
      devShells.aarch64-darwin.default = pkgs.mkShellNoCC {
        buildInputs = devinputs;
         };

      sing = pkgs.singularity-tools.buildImage {
        name = "equivariant-posteriors";
        diskSize = 1024 * 60;
        memSize = 1024 * 8;
        contents = 
          [ 
            (pkgs.buildEnv {
                  name = "root";
                  paths = [ pkgs.bashInteractive pkgs.coreutils program devinputs];
                  pathsToLink = [ "/bin" ];
                })          
              program
              devinputs
          ];
          # [ program (pkgs.python3.withPackages (p: [ p.numpy p.pytorch ])) ];
        runScript = ''
          #!${pkgs.stdenv.shell}
          exec /bin/sh $@'';
        runAsRoot =
          "   #!${pkgs.stdenv.shell}\n   ${pkgs.dockerTools.shadowSetup}\n";
      };

    };
}
