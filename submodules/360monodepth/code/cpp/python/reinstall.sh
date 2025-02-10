# !/bin/bash
python_vers=`python -V 2>&1`
split_vers=(${python_vers//./ })
compact_vers=${split_vers[1]}${split_vers[2]}
echo $compact_vers
echo "== 0) build project =="
cd ../build/
make -j 
cd ../python/
echo "== 1) remove build cache files =="
rm -rvf build/*
rm -rvf dist/*
# rm dist/instaOmniDepth-0.1.0-cp${compact_vers}-cp${compact_vers}-linux_x86_64.whl
echo "== 2) build python binding =="
python ./setup.py build
echo "== 3) build wheell package =="
python ./setup.py bdist_wheel
echo "== 4) reinstall package =="
pip uninstall --yes instaOmniDepth
pip install dist/instaOmniDepth-0.1.0-cp${compact_vers}-cp${compact_vers}-linux_x86_64.whl
