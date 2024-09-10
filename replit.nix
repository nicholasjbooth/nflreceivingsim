{ pkgs }: {
  deps = [
    pkgs.python311Full
    pkgs.glibcLocales
    pkgs.python3Packages.numpy
    pkgs.python3Packages.scipy
    pkgs.python3Packages.scikit-learn
    pkgs.python3Packages.pandas
  ];
}
