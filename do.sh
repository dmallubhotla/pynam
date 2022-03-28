#!/usr/bin/env bash
# Do - The Simplest Build Tool on Earth.
# Documentation and examples see https://github.com/8gears/do

set -Eeuo pipefail # -e "Automatic exit from bash shell script on error"  -u "Treat unset variables and parameters as errors"

build() {
	echo "I am ${FUNCNAME[0]}ing"
	poetry build
}

fmt() {
	poetry run black .
	find . -type f -name "*.py" -exec sed -i -e 's/    /\t/g' {} \;
}

test() {
	echo "I am ${FUNCNAME[0]}ing"
	poetry run flake8 pynam tests
	poetry run mypy pynam
	poetry run pytest
}

htmlcov() {
	poetry run pytest --cov-report=html
}

release() {
	./scripts/release.sh
}

all() {
	build && test
}

"$@" # <- execute the task

[ "$#" -gt 0 ] || printf "Usage:\n\t./do.sh %s\n" "($(compgen -A function | grep '^[^_]' | paste -sd '|' -))"
