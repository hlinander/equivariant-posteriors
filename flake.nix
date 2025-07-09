{
  inputs.nixpkgs.url = "github:cpcloud/nixpkgs/duckdb-1.3";
  inputs.helix-pkg.url = "github:helix-editor/helix";
  inputs.fenix = {
    url = "github:nix-community/fenix";
    inputs.nixpkgs.follows = "nixpkgs";
  };

  outputs = { self, nixpkgs, helix-pkg, fenix, ... }:
    let
      forSystem = system:
        let
          disableCudaEnvFlag = builtins.getEnv "DISABLE_CUDA";

          # --- FINAL OVERLAY ---
          # Disables test phase AND import checks for ibis packages.
          ibis-no-tests-overlay = final: prev: {
            python3Packages = prev.python3Packages.overrideScope (self: super: {
              ibis-framework = super.ibis-framework.overrideAttrs (old: {
                doCheck = false;
                doInstallCheck = false;
                pythonImportsCheck = []; # <-- ADD THIS LINE
              });
              ibis = super.ibis.overrideAttrs (old: {
                doCheck = false;
                doInstallCheck = false;
                pythonImportsCheck = []; # <-- AND THIS ONE
              });
            });
          };

          pkgs = import nixpkgs {
            inherit system;
            config = {
              allowUnfree = true;
              cudaSupport = disableCudaEnvFlag != "1";
            };
            overlays = [ ibis-no-tests-overlay ];
          };

          rustToolchain = fenix.packages."${system}".stable;
          helixmaster = helix-pkg.packages.${system}.default;

          weatherbench2 = pkgs.python3Packages.buildPythonPackage rec {
            pname = "weatherbench2";
            version = "0.2.0";
            format = "setuptools";
            src = pkgs.fetchFromGitHub {
              owner = "google-research";
              repo = pname;
              rev = "ae870189270ca46b96328a7328c877debf580ae8";
              hash = "sha256-2k83KxjWFANPwZWgrKwNstwbJjgtSDNZGDF0sPu2OVY=";
            };
            propagatedBuildInputs = [ ];
            doCheck = false;
          };

          healpix = pkgs.python3Packages.buildPythonPackage rec {
            pname = "healpix";
            version = "2023.1.13";
            format = "setuptools";
            src = pkgs.python3Packages.fetchPypi {
              inherit pname version;
              sha256 = "sha256-JU8AUgck7F8X2+0zk+O/F8iSt+laxt9S75p/a2jjhOE=";
            };
            propagatedBuildInputs = [ pkgs.python3Packages.numpy ];
            doCheck = false;
          };

          multiurl = pkgs.python3Packages.buildPythonPackage rec {
            pname = "multiurl";
            version = "0.3.1";
            format = "setuptools";
            src = pkgs.python3Packages.fetchPypi {
              inherit pname version;
              sha256 = "sha256-xwAUN7WdVtTDENclw9z/+YyXxLZSiT2ImJhTgnRl1EI=";
            };
            propagatedBuildInputs = [
              pkgs.python3Packages.requests
              pkgs.python3Packages.tqdm
              pkgs.python3Packages.pytz
              pkgs.python3Packages.dateutils
            ];
            doCheck = false;
          };

          cads_api_client = pkgs.python3Packages.buildPythonPackage rec {
            pname = "cads_api_client";
            version = "1.4.3";
            format = "pyproject";
            src = pkgs.python3Packages.fetchPypi {
              inherit pname version;
              sha256 = "sha256-UjWep0OoS1l8xYlzD/S0FHB7zk5eInmDVSiy0Sc8Gbg=";
            };
            propagatedBuildInputs = [
              pkgs.python3Packages.requests
              pkgs.python3Packages.tqdm
              pkgs.python3Packages.setuptools
              pkgs.python3Packages.setuptools_scm
              pkgs.python3Packages.attrs
              multiurl
            ];
            doCheck = false;
          };

          cdsapi = pkgs.python3Packages.buildPythonPackage rec {
            pname = "cdsapi";
            version = "0.7.3";
            format = "setuptools";
            src = pkgs.python3Packages.fetchPypi {
              inherit pname version;
              sha256 = "sha256-iDoTdspJVFfrVf1Ujbu29bZPLkyICzWG3Te6kEHlHII=";
            };
            propagatedBuildInputs = [
              pkgs.python3Packages.requests
              pkgs.python3Packages.tqdm
              cads_api_client
            ];
            doCheck = false;
          };

          eccodes3 = pkgs.python3Packages.buildPythonPackage rec {
            pname = "eccodes";
            version = "1.6.0";
            format = "setuptools";
            src = pkgs.python3Packages.fetchPypi {
              inherit pname version;
              sha256 = "sha256-WTkwQLz4nYBIEnQQkmWCqv/GiOF0vr1GRxkwe7xdnhU=";
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
            format = "setuptools";
            src = pkgs.python3Packages.fetchPypi {
              inherit pname version;
              sha256 = "sha256-wQgGBYyAxIYQwgG/BespJAGAeAarlCOrSWWuI9u2tSE=";
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
            format = "setuptools";
            src = pkgs.python3Packages.fetchPypi {
              inherit pname version;
              sha256 = "sha256-eoAVcemZ0O6D+bksu1mMIfhh7ibKnbp0zqiVi6QzXn4=";
            };
            propagatedBuildInputs = [ ];
            doCheck = false;
          };

          pythonPackages = with pkgs.python3Packages; [
            duckdb
            zarr
            weatherbench2
            flatbuffers
            psutil
            jupyter
            healpix
            cdsapi
            cfgrib
            netcdf4
            eccodes
            eccodes3
            xarray
            einops
            timm
            ipdb
            dill
            filelock
            gitpython
            tqdm
            python-lsp-server
            python-lsp-ruff
            numpy
            transformers
            peft
            datasets
            pytorch
            torchvision
            onnxruntime
            plotext
            torchmetrics
            ipython
            black
            flake8
            wandb
            snakeviz
            pandas
            matplotlib
            psycopg
            psycopg2
            pytest
            sqlalchemy
            (rpy2.override { extraRPackages = with pkgs.rPackages; [ ggplot2 ggforce dplyr latex2exp patchwork reticulate Hmisc ]; })
            seaborn
          ];

          pythonWithPackages = pkgs.python3.withPackages (p: pythonPackages);

          program = pkgs.python3Packages.buildPythonApplication {
            pname = "equivariant-transformers";
            version = "1.0";
            propagatedBuildInputs = pythonPackages;
            src = ./.;
            checkPhase = ''
              export TORCH_DEVICE="cpu"
              python test.py
            '';
          };

          devinputs = with pkgs; [
            helixmaster
            duckdb
            flatbuffers
            glslang
            shaderc
            linuxPackages_latest.perf
            glfw3
            glm
            xorg.libX11
            xorg.libpthreadstubs
            xorg.libXau
            xorg.libXdmcp
            xorg.libXrandr
            xorg.libXinerama
            xorg.libXcursor
            xorg.libXi
            xorg.libxcb
            xorg.xkbevd
            xorg.xkbutils
            xclip
            libxkbcommon
            libGL
            libGLU
            pkg-config
            openssl
            (rustToolchain.withComponents [
              "cargo"
              "rustc"
              "rust-src"
              "rustfmt"
              "clippy"
            ])
            fenix.packages."${system}".rust-analyzer
            nil
            julia
            postgresql_15
            ruff
            nixfmt-classic
            jq
            poppler
            fzf
            (ueberzugpp.override {
              enableWayland = false;
              enableOpencv = false;
            })
            (rWrapper.override {
              packages = with rPackages; [
                ggplot2
                dplyr
                latex2exp
                patchwork
                reticulate
                Hmisc
                RPostgreSQL
                plotly
              ];
            })
            cudatoolkit
            pythonWithPackages
          ];
        in
        {
          package = program;
          devShell =
            (pkgs.mkShell.override { stdenv = pkgs.llvmPackages_14.stdenv; }) {
              buildInputs = devinputs;
              nativeBuildInputs = [ pkgs.cudatoolkit ];
              shellHook = ''
                export POSTGRES=${pkgs.postgresql_15}
                export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/run/opengl-driver/lib/"
                export EDITOR=hx
                export PYTHON=${pythonWithPackages}/bin/python
                export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/run/opengl-driver/lib/:${
                  pkgs.lib.makeLibraryPath ([
                    pkgs.vulkan-loader
                    pkgs.libGL
                    pkgs.libGLU
                    pkgs.xorg.libX11
                    pkgs.xorg.libXcursor
                    pkgs.xorg.libXrandr
                    pkgs.xorg.libXi
                    pkgs.xorg.libxcb
                    pkgs.libxkbcommon
                    pkgs.pipewire
                    pkgs.openssl
                    pkgs.duckdb
                  ])
                }"
                export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:${pkgs.stdenv.cc.cc.lib}/lib64"
                export X11_X11_INCLUDE_PATH="${pkgs.xorg.libX11}/include"
                export X11_X11_LIB=${pkgs.lib.makeLibraryPath ([ pkgs.xorg.libX11 ])}
              '';
            };
          dbg = pkgs.python3.withPackages (p: [
            p.psycopg
            p.psycopg2
          ]);
          sing = pkgs.singularity-tools.buildImage {
            name = "equivariant-posteriors";
            diskSize = 1024 * 200;
            memSize = 1024 * 15;
            contents = [
              (pkgs.buildEnv {
                name = "root";
                paths = [
                  pkgs.bashInteractive
                  pkgs.coreutils
                  pkgs.findutils
                  pkgs.gnugrep
                ];
                pathsToLink = [ "/bin" "/share" ];
              })
              pkgs.postgresql_15
              pythonWithPackages
              pkgs.helix
              pkgs.gnugrep
              pkgs.findutils
              pkgs.vim
              pkgs.nix
              pkgs.glibcLocales
              pkgs.which
              pkgs.gnused
            ];
            runScript = ''
              #!${pkgs.stdenv.shell}
              export POSTGRES=${pkgs.postgresql_15}
              export LOCALE_ARCHIVE=${pkgs.glibcLocales}/lib/locale/locale-archive
              exec /bin/sh $@'';
            runAsRoot = "#!${pkgs.stdenv.shell}\n${pkgs.dockerTools.shadowSetup}\n";
          };
        };

      linuxSystem = "x86_64-linux";
      linuxOutputs = forSystem linuxSystem;

    in
    {
      devShells.${linuxSystem}.default = linuxOutputs.devShell;
      
      packages.${linuxSystem} = {
        default = linuxOutputs.package;
        sing = linuxOutputs.sing;
        dbg = linuxOutputs.dbg;
      };
    };
}
