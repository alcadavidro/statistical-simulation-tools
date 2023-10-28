generate-binary:
	python setup.py sdist bdist_wheel

install-package:
	pip install -e .
