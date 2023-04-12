{
  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
  outputs = { self, nixpkgs, ... }:
    let
      system = "x86_64-linux";
      pkgs = (import nixpkgs {
        inherit system;
        config = {
          allowUnfree = true;
          cudaSupport = true;
        };
      });
      program = pkgs.python3Packages.buildPythonApplication {
        pname = "equivariant-transformers";
        version = "1.0";

        propagatedBuildInputs = with pkgs.python3Packages; [ numpy pytorch ];
        # nativeCheckInputs = [ pkgs.cudatoolkit pkgs.linuxPackages.nvidia_x11 ];
        # checkInputs = [ pkgs.cudatoolkit pkgs.linuxPackages.nvidia_x11 ];

        src = ./.;

        checkPhase = ''
          #export CUDA_PATH="${pkgs.cudatoolkit}"
          #export LD_LIBRARY_PATH="${pkgs.linuxPackages.nvidia_x11}/lib"
          python main.py
        '';
      };
    in {
      packages.x86_64-linux.default = program;
      devShell.x86_64_linux = pkgs.mkShellNoCC {
        buildInputs = with pkgs; [
          helix
          nixfmt
          (python3.withPackages (p: [
            p.python-lsp-server
            p.numpy
            p.pytorch
            p.ipython
            p.black
            p.flake8
          ]))
        ];
      };

      sing = pkgs.singularity-tools.buildImage {
        name = "equivariant-posteriors";
        diskSize = 1024 * 60;
        memSize = 1024 * 8;
        contents =
          [ program (pkgs.python3.withPackages (p: [ p.numpy p.pytorch ])) ];
        runScript = ''
          #!${pkgs.stdenv.shell}
          exec /bin/sh $@'';
        runAsRoot =
          "   #!${pkgs.stdenv.shell}\n   ${pkgs.dockerTools.shadowSetup}\n";
      };

    };
}
