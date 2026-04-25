{ pkgs, lib, ... }:
let
  repoRoot = toString ./.;
  pythonEnv = pkgs.python312.withPackages (
    ps: (with ps; [
      setuptools
      wheel
      pip
      numpy
      pandas
      matplotlib
      scipy
      tifffile
      xarray
      pyyaml
      pillow
      imageio
      psutil
      dask
      pyarrow
      pyspark
      graphviz
      ipykernel
      pytest
      opencv4
    ]) ++ [
      ps."scikit-image"
      ps."scikit-learn"
      ps."dask-image"
      ps."pytest-cov"
    ]
  );
  pythonPath = lib.concatStringsSep ":" [
    "${repoRoot}/codeBase"
    "${repoRoot}/codeBase/image_utils"
    repoRoot
  ];
in {
  packages = [
    pythonEnv
    pkgs.graphviz
    pkgs.openjdk_headless
  ];

  env = {
    JAVA_HOME = "${pkgs.openjdk_headless}";
    PYTHON_BIN = "${pythonEnv}/bin/python";
    PYSPARK_PYTHON = "${pythonEnv}/bin/python";
    PYSPARK_DRIVER_PYTHON = "${pythonEnv}/bin/python";
    PYTHONPATH = pythonPath;
  };

  enterShell = ''
    echo "Lysozyme devenv ready."
    echo "  Python: $(python --version)"
    echo "  Try:    ./run.sh --help"
  '';

  enterTest = ''
    python - <<'PY'
import cv2
import dask
import image_utils
import imageio
import numpy
import pandas
import pipeline_implementations
import psutil
import pyspark
import scientific_image_finder
import tifffile
import xarray
import yaml
from graphviz import Digraph
from PIL import Image

print("Python imports OK")
PY

    python codeBase/run.py --help >/dev/null
    ./run.sh --help >/dev/null
  '';
}
