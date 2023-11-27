{
  # inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  # inputs.nixpkgs.url = "github:NixOS/nixpkgs/02f05fc";
  inputs.helix-pkg.url = "github:helix-editor/helix";
  # inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-22.11";
  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-23.11";
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
      datasets = pkgs.python3Packages.buildPythonPackage rec {
  pname = "datasets";
  version = "2.14.7";
  format = "setuptools";

  # disabled = pythonOlder "3.8";

  src = pkgs.fetchFromGitHub {
    owner = "huggingface";
    repo = pname;
    rev = "refs/tags/${version}";
    hash = "sha256-Q8cSgupfj6xKD0bYgL6bvYBwdYDdNaiWEWWUrRvwc4g";
  };

  propagatedBuildInputs = [
    pkgs.python3Packages.aiohttp
    pkgs.python3Packages.dill
    pkgs.python3Packages.fsspec
    pkgs.python3Packages.huggingface-hub
    pkgs.python3Packages.multiprocess
    pkgs.python3Packages.numpy
    pkgs.python3Packages.packaging
    pkgs.python3Packages.pandas
    pkgs.python3Packages.pyarrow
    pkgs.python3Packages.requests
    pkgs.python3Packages.responses
    pkgs.python3Packages.tqdm
    pkgs.python3Packages.xxhash
    pyarrow_hotfix
  ]; # ++ pkgs.lib.optionals (pkgs.pythonOlder "3.8") [
    #pkgs.python3Packages.importlib-metadata
  #];

  # Tests require pervasive internet access
  doCheck = false;

  # Module import will attempt to create a cache directory
  postFixup = "export HF_MODULES_CACHE=$TMPDIR";

  pythonImportsCheck = [
    "datasets"
  ];

  meta = with pkgs.lib; {
    description = "Open-access datasets and evaluation metrics for natural language processing";
    homepage = "https://github.com/huggingface/datasets";
    changelog = "https://github.com/huggingface/datasets/releases/tag/${version}";
    license = licenses.asl20;
    platforms = platforms.unix;
    maintainers = with maintainers; [ ];
  };
};
pyarrow_hotfix = pkgs.python3Packages.buildPythonPackage rec {
  pname = "pyarrow_hotfix";
  version = "1.0";

  src = ./pyarrow_hotfix; # Path to your local dummy_package directory
  format = "other";

  buildPhase = "runHook postBuild";

  installPhase = ''
    # mkdir -p $out/${pkgs.python3.sitePackages}
    # cp -r $src/* $out/${pkgs.python3.sitePackages}
    mkdir -p $out/${pkgs.python3.sitePackages}/pyarrow_hotfix
    cp -r $src/* $out/${pkgs.python3.sitePackages}/pyarrow_hotfix
  '';
  # No dependencies for this dummy package
  propagatedBuildInputs = [ ];

  # No tests to run
  doCheck = false;

  meta = with pkgs.lib; {
    description = "A dummy Python package";
    homepage = "https://example.com/dummy_package";
    license = licenses.mit;
  };
};
      
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
      # galgebra = pkgs.python3Packages.buildPythonPackage rec {
      #   pname = "galgebra";
      #   version = "0.5.0";
      #   src = pkgs.python3Packages.fetchPypi {
      #     inherit pname version;
      #     sha256 = "sha256-8Fb7DnIrdZw5lVF+e3oSeArASnraaKwzzmyA4PWBi6M="; # TODO
      #   };
      #   propagatedBuildInputs = [ pkgs.python3Packages.sympy  ];
      #   doCheck = false;
      # };
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
          # wandb
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
            # galgebra
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
            p.transformers
            p.peft
            datasets
            p.pytorch
            # (p.torchvision.override {torch = p.pytorch-bin;})
            p.torchvision
            # p.onnx
            p.onnxruntime
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
            #p.plotnine
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
