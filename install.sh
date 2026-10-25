#!/usr/bin/env bash
# Install or update investalyze as an app, apart from this repo: one version from GitHub in a Python environment in the app
# folder, and the settings files on the first install. An update replaces only the environment.
# Usage: ./install.sh <version>                                for example ./install.sh v0.5.0
#        INVESTALYZE_HOME=/other/app/folder ./install.sh <version>
set -euo pipefail

VERSION=${1:?Give the version to install, for example: ./install.sh v0.5.0}
APP=${INVESTALYZE_HOME:-$HOME/.local/share/md_apps/investalyze}
REPO=marcdumon/investalyze

for tool in uv curl; do
    command -v "$tool" >/dev/null || { echo "Missing '$tool': install it first." >&2; exit 1; }
done

mkdir -p "$APP"
[[ -d $APP/env ]] || uv venv --quiet --python 3.13 "$APP/env"
echo "Installing investalyze $VERSION in $APP/env ..."
# torch comes from the CUDA 12.1 index that pyproject.toml names (see the note there)
uv pip install --quiet --python "$APP/env/bin/python" --reinstall-package investalyze "git+https://github.com/$REPO@$VERSION"

for name in investalyze.example ingest backtest cleaning stooq_tickers; do
    target=$APP/${name%.example}.toml
    if [[ ! -f $target ]]; then
        curl -sfL "https://raw.githubusercontent.com/$REPO/$VERSION/$name.toml" -o "$target"
        echo "Made $target: set your own values in it."
    fi
done
echo "Installed. Start the app with: $APP/env/bin/investalyze"
