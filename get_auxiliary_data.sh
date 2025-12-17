# Downloads data needed for data augmentation and training on auxiliary data.

# Political boundaries
curl -L --create-dirs \
  -o data/geocells/geoBoundariesCGAZ_ADM2.geojson \
  https://github.com/wmgeolab/geoBoundaries/raw/main/releaseData/CGAZ/geoBoundariesCGAZ_ADM2.geojson

curl -L --create-dirs -O --output-dir data/geocells https://github.com/wmgeolab/geoBoundaries/raw/main/releaseData/CGAZ/geoBoundariesCGAZ_ADM2.geojson

# GADM Country Area Data
curl --create-dirs -O --output-dir data/gadm https://geodata.ucdavis.edu/gadm/gadm4.1/gadm_410-levels.zip
cd data/gadm
unzip gadm_410-levels.zip
cd ../..