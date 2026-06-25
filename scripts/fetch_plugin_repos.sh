#!/usr/bin/env bash
set -euo pipefail

mkdir -p external/plugin_baselines

clone_or_update() {
  local name="$1"
  local url="$2"
  local commit="$3"
  local dst="external/plugin_baselines/${name}"
  if [[ -d "${dst}/.git" ]]; then
    git -C "${dst}" fetch --all --tags
    git -C "${dst}" checkout "${commit}"
    return 0
  fi
  if [[ -d "${dst}" ]]; then
    echo "${dst} exists as a vendored directory without .git; using the checked-in copy."
    echo "To refresh from upstream, remove this directory first or use a fresh checkout."
    return 0
  fi
  if [[ ! -d "${dst}/.git" ]]; then
    git clone "${url}" "${dst}"
  fi
  git -C "${dst}" fetch --all --tags
  git -C "${dst}" checkout "${commit}"
}

clone_or_update RevIN https://github.com/ts-kim/RevIN.git fee40bc6c87cb536d048bcf1c14c4ed644b875e1
clone_or_update Dish-TS https://github.com/weifantt/Dish-TS.git e674d3b94b832491f63a533d60e40a75031d2c75
clone_or_update SAN https://github.com/icantnamemyself/SAN.git 7e1ca66251a91a89290846b310145c5f5db3ffc3
clone_or_update Nonstationary_Transformers https://github.com/thuml/Nonstationary_Transformers.git c4ec40675d11d50b3d9923657f408d0db6f90f56
clone_or_update TAFAS https://github.com/kimanki/TAFAS.git 139bf980671da4daad728a0fc21d8df508b9203d

echo "Official same-position baseline repos fetched under external/plugin_baselines/."
echo "Pinned source identities are recorded in docs/core_table_manifest.yaml and docs/baseline_repos.tsv."
