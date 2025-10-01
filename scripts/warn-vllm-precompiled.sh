#!/usr/bin/env bash
set -euo pipefail

# Use env var if present; otherwise allow passing as $1
VLLM_COMMIT_SHA="${VLLM_COMMIT_SHA:-unknown}"

append_warning() {
  local target="$1"
  mkdir -p "$(dirname "$target")"
  touch "$target"

  if ! grep -q 'BEGIN vllm precompiled warning' "$target"; then
    cat >>"$target" <<EOF

# ----- BEGIN vllm precompiled warning -----
# Generated at: $(date -u +'%Y-%m-%dT%H:%M:%SZ')
vllm_precompiled_warning() {
  printf '%s\n' "Warning! vLLM has been installed editably at path /opt/vllm-source but with precompiled binaries for sha: ${VLLM_COMMIT_SHA}!"
  printf '%s\n' "If you want to swap your commit SHA for development, uninstall the wheel and reinstall from source:"
  cat <<'SH'
uv pip uninstall vllm
unset VLLM_USE_PRECOMPILED VLLM_PRECOMPILED_WHEEL_LOCATION
# git -C /opt/vllm-source remote add myfork <your_fork>   # optionally add your fork
# git -C /opt/vllm-source fetch myfork                    # optionally fetch from your fork
# git -C /opt/vllm-source checkout <your_commit>
# reinstall from source:
uv pip install -e /opt/vllm-source --no-cache-dir
SH
}

# Auto-run only for interactive shells
case \$- in *i*) vllm_precompiled_warning ;; esac
# ----- END vllm precompiled warning -----

EOF
  fi
}

append_warning /etc/bashrc

# Append to vllm user's rc if it exists; otherwise just create it
if [ -f /home/vllm/.bashrc ]; then
  append_warning /home/vllm/.bashrc
  chown vllm:root /home/vllm/.bashrc || true
fi
