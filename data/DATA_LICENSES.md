# Public dataset attribution

The datasets are provided under Creative Commons Attribution 4.0 International (CC BY 4.0), as stated on their source repository pages, checked on 10 October 2026. The software MIT license applies separately to the experiment implementation.

| Dataset | Source and persistent identifier |
| --- | --- |
| Dry Bean | [UCI source](https://archive.ics.uci.edu/dataset/602/dry+bean+dataset); [dataset DOI](https://doi.org/10.24432/C50S4B) |
| Letter Recognition | [UCI source](https://archive.ics.uci.edu/dataset/59/letter+recognition); [dataset DOI](https://doi.org/10.24432/C5ZP40) |
| Optical Recognition of Handwritten Digits | [UCI source](https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits); [dataset DOI](https://doi.org/10.24432/C50P49) |
| Pen-Based Recognition of Handwritten Digits | [UCI source](https://archive.ics.uci.edu/dataset/81/pen+based+recognition+of+handwritten+digits); [dataset DOI](https://doi.org/10.24432/C5MG6K) |
| Statlog (Landsat Satellite) | [UCI source](https://archive.ics.uci.edu/dataset/146/statlog+landsat+satellite); [dataset DOI](https://doi.org/10.24432/C55887) |
| Dataset for Sensorless Drive Diagnosis | [UCI source](https://archive.ics.uci.edu/dataset/325/dataset+for+sensorless+drive+diagnosis); [dataset DOI](https://doi.org/10.24432/C5VP5F) |
| Statlog (Shuttle) | [UCI source](https://archive.ics.uci.edu/dataset/148/statlog+shuttle); [dataset DOI](https://doi.org/10.24432/C5WS31) |
| Waveform Database Generator (Version 1) | [UCI source](https://archive.ics.uci.edu/dataset/107/waveform+database+generator+version+1); [dataset DOI](https://doi.org/10.24432/C5CS3C) |

License: https://creativecommons.org/licenses/by/4.0/

## Processing

Numerical CSV data are retained with their original observations. Sensorless Drive Diagnosis is converted from its source text file into CSV with 48 feature columns and a class column. The Shuttle download has an inconsistent header: eight names accompany ten fields per row. The processed CSV assigns nine neutral feature names and retains the final class field, preserving all original numeric values. Training uses stratified splits; numeric predictors are standardized using training records only. Labels are encoded for modeling. Frozen score populations include original row indices and model-derived prediction cells.

Source creators recorded by UCI: David Slate (Letter); Martyna Bator (Sensorless); E. Alpaydin and Fevzi Alimoglu (Pen Digits); E. Alpaydin and C. Kaynak (Optical Digits); Ashwin Srinivasan (Satellite); L. Breiman and C. J. Stone (Waveform). Dry Bean is associated with the data study by M. Koklu and I. A. Ozkan, DOI 10.1016/j.compag.2020.105507. The Shuttle repository record does not name an individual creator.

`manifest.json` records the exact input hashes and download URLs.
