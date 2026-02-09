let
  pkgs = import <nixpkgs> {};
in

pkgs.mkShell {
  packages = with pkgs; [
    (python3.withPackages (pypkgs: with pypkgs; [
      llm
      llm-gemini
      llm-perplexity
      llm-mistral
      llm-grok
      llm-anthropic
    ]))
  ];

  shellHook = ''
    export PS1="[\[\033[01;32m\]nix-shell\[\033[00m\]:\W] \[\033[01;32m\]λ\[\033[00m\] "
  '';
}
