# Attributions

This document lists all third-party libraries, packages, and tools used in the lysozyme stain quantification pipeline, along with their attributions and licensing information.

## Python Dependencies

### Core Scientific Computing Libraries

**NumPy**
- Purpose: Fundamental package for array operations and numerical computing
- Website: https://numpy.org/
- License: BSD-3-Clause
- Citation: Harris, C.R., Millman, K.J., van der Walt, S.J. et al. Array programming with NumPy. Nature 585, 357–362 (2020).


@Article{         harris2020array,
 title         = {Array programming with {NumPy}},
 author        = {Charles R. Harris and K. Jarrod Millman and St{\'{e}}fan J.
                 van der Walt and Ralf Gommers and Pauli Virtanen and David
                 Cournapeau and Eric Wieser and Julian Taylor and Sebastian
                 Berg and Nathaniel J. Smith and Robert Kern and Matti Picus
                 and Stephan Hoyer and Marten H. van Kerkwijk and Matthew
                 Brett and Allan Haldane and Jaime Fern{\'{a}}ndez del
                 R{\'{i}}o and Mark Wiebe and Pearu Peterson and Pierre
                 G{\'{e}}rard-Marchant and Kevin Sheppard and Tyler Reddy and
                 Warren Weckesser and Hameer Abbasi and Christoph Gohlke and
                 Travis E. Oliphant},
 year          = {2020},
 month         = sep,
 journal       = {Nature},
 volume        = {585},
 number        = {7825},
 pages         = {357--362},
 doi           = {10.1038/s41586-020-2649-2},
 publisher     = {Springer Science and Business Media {LLC}},
 url           = {https://doi.org/10.1038/s41586-020-2649-2}
}

**SciPy**
- Purpose: Scientific computing library for optimization, linear algebra, and signal processing
- Website: https://scipy.org/
- License: BSD-3-Clause
- Citation: Virtanen, P., Gommers, R., Oliphant, T.E. et al. SciPy 1.0: fundamental algorithms for scientific computing in Python. Nat Methods 17, 261–272 (2020).


@ARTICLE{2020SciPy-NMeth,
  author  = {Virtanen, Pauli and Gommers, Ralf and Oliphant, Travis E. and
            Haberland, Matt and Reddy, Tyler and Cournapeau, David and
            Burovski, Evgeni and Peterson, Pearu and Weckesser, Warren and
            Bright, Jonathan and {van der Walt}, St{\'e}fan J. and
            Brett, Matthew and Wilson, Joshua and Millman, K. Jarrod and
            Mayorov, Nikolay and Nelson, Andrew R. J. and Jones, Eric and
            Kern, Robert and Larson, Eric and Carey, C J and
            Polat, {\.I}lhan and Feng, Yu and Moore, Eric W. and
            {VanderPlas}, Jake and Laxalde, Denis and Perktold, Josef and
            Cimrman, Robert and Henriksen, Ian and Quintero, E. A. and
            Harris, Charles R. and Archibald, Anne M. and
            Ribeiro, Ant{\^o}nio H. and Pedregosa, Fabian and
            {van Mulbregt}, Paul and {SciPy 1.0 Contributors}},
  title   = {{{SciPy} 1.0: Fundamental Algorithms for Scientific
            Computing in Python}},
  journal = {Nature Methods},
  year    = {2020},
  volume  = {17},
  pages   = {261--272},
  adsurl  = {https://rdcu.be/b08Wh},
  doi     = {10.1038/s41592-019-0686-2},
}

**scikit-image**
- Purpose: Image processing library for segmentation, morphological operations, and analysis
- Website: https://scikit-image.org/
- License: BSD-3-Clause
- Citation: van der Walt, S., Schönberger, J.L., Nunez-Iglesias, J. et al. scikit-image: image processing in Python. PeerJ 2:e453 (2014).

@article{scikit-image,
 title = {scikit-image: image processing in {P}ython},
 author = {van der Walt, {S}t\'efan and {S}ch\"onberger, {J}ohannes {L}. and
           {Nunez-Iglesias}, {J}uan and {B}oulogne, {F}ran\c{c}ois and {W}arner,
           {J}oshua {D}. and {Y}ager, {N}eil and {G}ouillart, {E}mmanuelle and
           {Y}u, {T}ony and the scikit-image contributors},
 year = {2014},
 month = {6},
 keywords = {Image processing, Reproducible research, Education,
             Visualization, Open source, Python, Scientific programming},
 volume = {2},
 pages = {e453},
 journal = {PeerJ},
 issn = {2167-8359},
 url = {https://doi.org/10.7717/peerj.453},
 doi = {10.7717/peerj.453}
}



**scikit-learn**
- Purpose: Machine learning library used for linear regression in scoring algorithms
- Website: https://scikit-learn.org/
- License: BSD-3-Clause
- Citation: Pedregosa, F. et al. Scikit-learn: Machine Learning in Python. JMLR 12, pp. 2825-2830 (2011).

**pandas**
- Purpose: Data manipulation and analysis library for handling summary statistics
- Website: https://pandas.pydata.org/
- License: BSD-3-Clause
- Citation: McKinney, W. Data Structures for Statistical Computing in Python, Proceedings of the 9th Python in Science Conference, 51-56 (2010).

### Image I/O and Visualization

**tifffile**
- Purpose: Read and write TIFF image files, particularly microscopy images
- Website: https://github.com/cgohlke/tifffile
- License: BSD-3-Clause
- Author: Christoph Gohlke

**matplotlib**
- Purpose: Plotting library for data visualization and image display
- Website: https://matplotlib.org/
- License: PSF-based (Python Software Foundation License)
- Citation: Hunter, J.D. Matplotlib: A 2D graphics environment. Computing in Science & Engineering 9, 90-95 (2007).

**OpenCV (opencv-python-headless)**
- Purpose: Computer vision library for image processing, contour detection, and thresholding
- Website: https://opencv.org/
- License: Apache License 2.0
- Citation: Bradski, G. The OpenCV Library. Dr. Dobb's Journal of Software Tools (2000).

### Specialized Scientific Libraries

**Shapely**
- Purpose: Geometric operations and spatial analysis for contour processing
- Website: https://shapely.readthedocs.io/
- License: BSD-3-Clause
- Citation: Gillies, S. et al. Shapely: manipulation and analysis of geometric objects. https://github.com/Toblerity/Shapely

**roifile**
- Purpose: Read and write ImageJ ROI files for integration with ImageJ/FIJI
- Website: https://github.com/cgohlke/roifile
- License: BSD-3-Clause
- Author: Christoph Gohlke

### Development and Testing

**pytest**
- Purpose: Testing framework for unit and integration tests
- Website: https://pytest.org/
- License: MIT License
- Citation: Krekel, H. et al. pytest: helps you write better programs. https://github.com/pytest-dev/pytest

**Jupyter**
- Purpose: Interactive notebook environment for development and analysis
- Website: https://jupyter.org/
- License: BSD-3-Clause
- Citation: Kluyver, T. et al. Jupyter Notebooks – a publishing format for reproducible computational workflows. Positioning and Power in Academic Publishing: Players, Agents and Agendas, pp. 87-90 (2016).

### Python Standard Library Modules

The following standard library modules are used extensively:
- `pathlib` - Object-oriented filesystem paths
- `collections` - Specialized container datatypes (defaultdict, Counter)
- `itertools` - Iterator creation functions
- `concurrent.futures` - High-level interface for asynchronously executing callables
- `json` - JSON encoder and decoder
- `re` - Regular expression operations
- `os` - Operating system interface
- `sys` - System-specific parameters and functions
- `tempfile` - Generate temporary files and directories
- `time` - Time-related functions
- `typing` - Support for type hints
- `dataclasses` - Classes for storing data
- `textwrap` - Text wrapping utilities
- `zipfile` - Work with ZIP archives
- `unittest` - Unit testing framework
- `atexit` - Exit handlers
- `gc` - Garbage collection interface

## R Dependencies

### Data Manipulation and Analysis

**dplyr**
- Purpose: Grammar of data manipulation
- Website: https://dplyr.tidyverse.org/
- License: MIT License
- Citation: Wickham, H. et al. dplyr: A Grammar of Data Manipulation. R package version 1.0.0.

**tidyr**
- Purpose: Tidy messy data
- Website: https://tidyr.tidyverse.org/
- License: MIT License
- Citation: Wickham, H. and Henry, L. tidyr: Tidy Messy Data. R package version 1.1.0.

**readr**
- Purpose: Read rectangular text data
- Website: https://readr.tidyverse.org/
- License: MIT License
- Citation: Wickham, H. et al. readr: Read Rectangular Text Data. R package version 2.0.0.

**stringr**
- Purpose: Simple, consistent wrappers for common string operations
- Website: https://stringr.tidyverse.org/
- License: MIT License
- Citation: Wickham, H. stringr: Simple, Consistent Wrappers for Common String Operations. R package version 1.4.0.

### Visualization

**ggplot2**
- Purpose: Create elegant data visualizations using the grammar of graphics
- Website: https://ggplot2.tidyverse.org/
- License: MIT License
- Citation: Wickham, H. ggplot2: Elegant Graphics for Data Analysis. Springer-Verlag New York, 2016.

**RColorBrewer**
- Purpose: ColorBrewer palettes for R
- Website: https://cran.r-project.org/package=RColorBrewer
- License: Apache License 2.0
- Citation: Neuwirth, E. RColorBrewer: ColorBrewer Palettes. R package version 1.1-2.

## External Tools and Integration

### Image Analysis Software

**ImageJ/FIJI**
- Purpose: ROI export compatibility through roifile library
- Website: https://imagej.nih.gov/ij/
- License: Public Domain
- Citation: Schneider, C.A., Rasband, W.S., Eliceiri, K.W. NIH Image to ImageJ: 25 years of image analysis. Nature Methods 9, 671-675 (2012).

**QuPath**
- Purpose: GeoJSON export compatibility for pathology image analysis
- Website: https://qupath.github.io/
- License: GPLv3
- Citation: Bankhead, P. et al. QuPath: Open source software for digital pathology image analysis. Scientific Reports 7, 16878 (2017).

### High-Performance Computing

**SLURM Workload Manager**
- Purpose: Job scheduling and resource management for cluster computing
- Website: https://slurm.schedmd.com/
- License: GPLv2
- Citation: Yoo, A.B., Jette, M.A., Grondona, M. SLURM: Simple Linux Utility for Resource Management. Job Scheduling Strategies for Parallel Processing, pp. 44-60 (2003).

## System Dependencies

- **Python 3.8+** - Primary programming language
- **R 4.0+** - Statistical computing environment
- **Bash shell** - Command-line scripting
- **Linux/Unix environment** - Primary operating system target

## Methodology References

### Image Processing Algorithms

**Watershed Segmentation**
- Beucher, S., Lantuéjoul, C. Use of watersheds in contour detection. International workshop on image processing: real-time edge and motion detection/estimation (1979).

**Morphological Operations**
- Serra, J. Image Analysis and Mathematical Morphology. Academic Press (1982).

**Connected Component Analysis**
- Rosenfeld, A., Pfaltz, J.L. Sequential operations in digital picture processing. Journal of the ACM 13, 471-494 (1966).

### Statistical Methods

**Linear Regression**
- Galton, F. Regression towards mediocrity in hereditary stature. Journal of the Anthropological Institute of Great Britain and Ireland 15, 246-263 (1886).

**Outlier Detection**
- Tukey, J.W. Exploratory Data Analysis. Addison-Wesley (1977).

## License Compatibility

This project combines libraries under various open-source licenses:
- **BSD-3-Clause**: NumPy, SciPy, scikit-image, scikit-learn, pandas, tifffile, Shapely, roifile, Jupyter
- **MIT License**: pytest, R tidyverse packages (dplyr, tidyr, readr, stringr, ggplot2)
- **Apache License 2.0**: OpenCV, RColorBrewer
- **PSF License**: matplotlib
- **GPLv2/GPLv3**: SLURM, QuPath (integration only)
- **Public Domain**: ImageJ (integration only)

All licenses are compatible with open-source research use. For commercial applications, please review individual package licenses.

## Version Information

This attribution document is current as of September 2025. Package versions and licensing may change over time. For the most current version information, see `requirements.txt` and individual package documentation.

## Contributing

When adding new dependencies to this project, please:
1. Add the package to the appropriate requirements file
2. Update this attributions document with proper citation information
3. Verify license compatibility with existing dependencies
4. Test that the new dependency integrates properly with the existing pipeline

## Acknowledgments

We acknowledge the scientific Python and R communities for developing and maintaining these excellent open-source tools that make reproducible research possible.
