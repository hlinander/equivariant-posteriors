{
  inputs.nixpkgs.url = "github:NixOS/nixpkgs";
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
      healpix = pkgs.python3Packages.buildPythonPackage rec {
        pname = "healpix";
        version = "2023.1.13";
        src = pkgs.python3Packages.fetchPypi {
          inherit pname version;
          sha256 = "sha256-JU8AUgck7F8X2+0zk+O/F8iSt+laxt9S75p/a2jjhOE="; # TODO
        };
        propagatedBuildInputs = [pkgs.python3Packages.numpy];
        doCheck = false;
      };
       cdsapi = pkgs.python3Packages.buildPythonPackage rec {
        pname = "cdsapi";
        version = "0.6.1";
        src = pkgs.python3Packages.fetchPypi {
          inherit pname version;
          sha256 = "sha256-fUDFjj/T51qKzc3IHqtO+bb3Y7KQK6AdfRc482UqWjA="; # TODO
        };
        propagatedBuildInputs = [ 
         pkgs.python3Packages.requests
         pkgs.python3Packages.tqdm
         
        ];
        doCheck = false;
      };
      eccodes3 = pkgs.python3Packages.buildPythonPackage rec {
        pname = "eccodes";
        version = "1.6.0";
        src = pkgs.python3Packages.fetchPypi {
          inherit pname version;
          sha256 = "sha256-WTkwQLz4nYBIEnQQkmWCqv/GiOF0vr1GRxkwe7xdnhU="; # TODO
        };
        propagatedBuildInputs = [ 
         pkgs.python3Packages.numpy
         pkgs.eccodes
         pkgs.python3Packages.cffi
         pkgs.python3Packages.attrs
         findlibs
        ];
        doCheck = false;
      };
      cfgrib = pkgs.python3Packages.buildPythonPackage rec {
        pname = "cfgrib";
        version = "0.9.10.3";
        src = pkgs.python3Packages.fetchPypi {
          inherit pname version;
          sha256 = "sha256-wQgGBYyAxIYQwgG/BespJAGAeAarlCOrSWWuI9u2tSE="; # TODO
        };
        propagatedBuildInputs = [ 
         pkgs.python3Packages.numpy
         eccodes3
         pkgs.python3Packages.click
         pkgs.python3Packages.attrs
         pkgs.python3Packages.xarray
        ];
        doCheck = false;
      };
      findlibs = pkgs.python3Packages.buildPythonPackage rec {
        pname = "findlibs";
        version = "0.0.5";
        src = pkgs.python3Packages.fetchPypi {
          inherit pname version;
          sha256 = "sha256-eoAVcemZ0O6D+bksu1mMIfhh7ibKnbp0zqiVi6QzXn4="; # TODO
        };
        propagatedBuildInputs = [  ];
        doCheck = false;
      };
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

        propagatedBuildInputs = with pkgs.python3Packages; [
          tqdm
          numpy
          pytorch
          torchvision
          torchmetrics
          plotext
          wandb
          pandas
          psycopg
          pytest
          filelock
          einops
          timm
          healpix
        ];

        src = ./.;

        checkPhase = ''
          # ${pkgs.python3Packages.pytorch}/bin/torchrun --standalone --nnodes=1 --nproc_per_node=1 test.py
          export TORCH_DEVICE="cpu"
          python test.py
        '';
      };

      pythonWithPackages = with pkgs; (python3.withPackages (p: [
            (p.rpy2.override{ extraRPackages = with rPackages; [ggplot2 dplyr latex2exp patchwork reticulate Hmisc]; })
            p.jupyter
            healpix
            # p.torchUncertainty
            cdsapi
            cfgrib
            p.netcdf4
            p.eccodes
            eccodes3
            p.xarray
            p.einops
            p.timm
            p.ipdb
            p.dill
            p.filelock
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
            p.psycopg2
            p.pytest
            p.sqlalchemy
          ]));
      devinputs = with pkgs; [
          julia
          postgresql
          ruff
          helixmaster
          nixfmt
          jq
          yazi
          nerdfonts
          poppler
          fzf
          (ueberzugpp.override { enableWayland=false; enableOpencv=false; })
          (rWrapper.override{ packages = with rPackages; [ ggplot2 dplyr latex2exp patchwork reticulate Hmisc RPostgreSQL plotly]; })
          (rstudioWrapper.override{ packages = with rPackages; [ ggplot2 dplyr patchwork reticulate Hmisc RPostgreSQL plotly esquisse matlab ggExtra ggpubr]; })
          # cudatoolkit
          pythonWithPackages
        ];
    in {
      packages.x86_64-linux.default = program;
      packages.aarch64-darwin.default = program;
      devShells.x86_64-linux.default = pkgs.mkShellNoCC {
        buildInputs = devinputs;
        nativeBuildInputs = [pkgs.cudatoolkit];
        shellHook = '' 
          export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/run/opengl-driver/lib/"
          export EDITOR=hx
          export PYTHON=${pythonWithPackages}/bin/python
          # export CUDA_PATH=${pkgs.cudatoolkit}
        '';
         };
      devShells.aarch64-darwin.default = pkgs.mkShellNoCC {
        buildInputs = devinputs;
         };

      sing = pkgs.singularity-tools.buildImage {
        name = "equivariant-posteriors";
        diskSize = 1024 * 100;
        memSize = 1024 * 8;
        contents = 
          [ 
            (pkgs.buildEnv {
                  name = "root";
                  paths = [ pkgs.bashInteractive pkgs.coreutils pkgs.findutils pkgs.gnugrep program devinputs];
                  pathsToLink = [ "/bin" ];
                })          
              program
              devinputs
              pkgs.gnugrep
              pkgs.findutils
              pkgs.vim
              pkgs.nix
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
