rmdir /S /Q build
del dist/instaOmniDepth-0.1.0-cp38-cp38-win_amd64.whl
python ./setup.py build
python ./setup.py bdist_wheel
pip uninstall --yes instaOmniDepth
pip install dist/instaOmniDepth-0.1.0-cp38-cp38-win_amd64.whl