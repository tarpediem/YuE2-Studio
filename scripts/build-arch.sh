#!/bin/bash
# Native Arch Linux build of YuE2 Studio with CUDA, installed for the current
# user only (no root, no package): the studio, the pinned yue2.cpp engine on
# CUDA + Vulkan, and HOT-Step's MIDI transcriber, trainer and captioner on CUDA.
# The Linux code looks for the tools in <executable folder>/resources/<tool>/,
# so the install is laid out that way and never marked portable (data stays
# in the XDG directories).
#
# usage: build-arch.sh [INSTALL_DIR]     (default ~/.local/opt/yue2-studio-native)
#
# Environment:
#   YUE_BUILD_CACHE  clones, build trees and the cargo target (default ~/.cache/yue2-build);
#                    builds are incremental, so a second run only rebuilds what changed
#   YUE_CUDA_ARCH    CMAKE_CUDA_ARCHITECTURES (default 86-real, the RTX 30 series).
#                    `native` would ask nvcc for the card of the build machine, but with
#                    GGML_NATIVE off ggml sets no architecture of its own, so say it
#                    explicitly: e.g. 89-real for an RTX 40, "86-real;89-real" for both.
#   NVCC_CCBIN       host compiler nvcc must use (default: the one /etc/profile.d/cuda.sh
#                    exports; CUDA 13 rejects an Arch gcc that is too recent)
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
install_dir="${1:-$HOME/.local/opt/yue2-studio-native}"
cache="${YUE_BUILD_CACHE:-$HOME/.cache/yue2-build}"
cuda_arch="${YUE_CUDA_ARCH:-86-real}"
jobs="$(getconf _NPROCESSORS_ONLN)"
# heavy builds must not freeze the desktop
run() { nice -n 15 "$@"; }

# ---- prerequisites: report everything that is missing at once ----------------
# `pacman -T` also honours "provides", so onnxruntime-cpu/-cuda satisfy onnxruntime.
missing="$(pacman -T cuda webkit2gtk-4.1 gst-plugins-base gst-plugins-good vulkan-headers spirv-headers shaderc cmake ninja nodejs npm git || true)"
command -v glslc >/dev/null || missing="$missing"$'\n'"shaderc (glslc)"
command -v cargo >/dev/null || command -v rustup >/dev/null || missing="$missing"$'\n'"rustup"
if [ -n "${missing//[$'\n ']/}" ]; then
    echo "[ERROR] missing Arch packages, install them first:" >&2
    echo "$missing" | sed '/^$/d; s/^/    /' >&2
    echo "    sudo pacman -S --needed cuda webkit2gtk-4.1 gst-plugins-base gst-plugins-good vulkan-headers spirv-headers shaderc cmake ninja nodejs npm rustup" >&2
    exit 1
fi
[ -z "$(pacman -T onnxruntime || true)" ] \
    || echo "[WARN] onnxruntime is not installed: the studio's ONNX features will not run (sudo pacman -S onnxruntime)" >&2

# ---- CUDA environment ----------------------------------------------------------
export CUDA_PATH=/opt/cuda
export PATH="/opt/cuda/bin:$PATH"
if [ -z "${NVCC_CCBIN:-}" ] && [ -r /etc/profile.d/cuda.sh ]; then
    NVCC_CCBIN="$(sed -n "s/^export NVCC_CCBIN='\{0,1\}\([^']*\)'\{0,1\}\$/\1/p" /etc/profile.d/cuda.sh | tail -n 1)"
fi
[ -x "${NVCC_CCBIN:-}" ] || { echo "[ERROR] no usable host compiler for nvcc: set NVCC_CCBIN (e.g. /usr/bin/g++-15)" >&2; exit 1; }
command -v nvcc >/dev/null || { echo "[ERROR] nvcc not found in /opt/cuda/bin" >&2; exit 1; }
echo "CUDA architectures: $cuda_arch, host compiler: $NVCC_CCBIN"

rpath_flags=(-DCMAKE_BUILD_RPATH='$ORIGIN' -DCMAKE_INSTALL_RPATH='$ORIGIN' -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON)
cuda_flags=("-DCMAKE_CUDA_ARCHITECTURES=$cuda_arch" "-DCMAKE_CUDA_HOST_COMPILER=$NVCC_CCBIN")

json() { sed -n "s/.*\"$2\": \"\(.*\)\".*/\1/p" "$root/engines/$1"; }

# The ggml backends are modules nothing links against, so each processor
# variant is a ninja target of its own: read them from the configured build.
cpu_variants() {
    local found
    found="$(sed -n 's/^build \(ggml-cpu-[a-z0-9_]*\): phony.*/\1/p' "$1/build.ninja" | sort -u | tr '\n' ' ')"
    [ -n "$found" ] || { echo "[ERROR] the configured build in $1 has no processor variants of ggml-cpu" >&2; exit 1; }
    echo "$found"
}

mkdir -p "$cache"

# ---- yue2.cpp ---------------------------------------------------------------------
yue_commit="$(json yue2-cpp-source.json commit)"
yue_tree="$cache/yue2-engine"
[ -d "$yue_tree/.git" ] || git clone "$(json yue2-cpp-source.json repository)" "$yue_tree"
yue_changed=0
if [ "$(git -C "$yue_tree" rev-parse HEAD)" != "$yue_commit" ]; then
    git -C "$yue_tree" fetch --quiet origin "$yue_commit" || git -C "$yue_tree" fetch --quiet origin
    git -C "$yue_tree" checkout --quiet "$yue_commit"
    yue_changed=1
fi
git -C "$yue_tree" submodule update --init --recursive
yue_build="$yue_tree/build-cuda"
if [ "$yue_changed" = 1 ] || [ ! -f "$yue_build/CMakeCache.txt" ]; then
    run cmake -S "$yue_tree" -B "$yue_build" -G Ninja -DCMAKE_BUILD_TYPE=Release \
        -DGGML_NATIVE=OFF -DGGML_CUDA=ON -DGGML_VULKAN=ON -DGGML_BACKEND_DL=ON -DGGML_CPU_ALL_VARIANTS=ON \
        "${cuda_flags[@]}" "${rpath_flags[@]}"
fi
run cmake --build "$yue_build" --target yue-server ggml-cuda ggml-vulkan $(cpu_variants "$yue_build") -j "$jobs"

# ---- HOT-Step-CPP: MIDI transcriber and trainer, each at its own commit ------------
hotstep="$cache/HOT-Step-CPP"
[ -d "$hotstep/.git" ] || git clone "$(json music-midi-source.json repository)" "$hotstep"

# $1 worktree name, $2 engines/*.json, $3... ninja targets. Prints nothing: the tree is $cache/$1.
build_hotstep() {
    local name="$1" source_file="$2" commit tree build
    shift 2
    commit="$(json "$source_file" commit)"
    tree="$cache/$name"
    if [ ! -e "$tree/.git" ]; then
        git -C "$hotstep" fetch --quiet origin "$commit" || git -C "$hotstep" fetch --quiet origin
        git -C "$hotstep" worktree add --detach "$tree" "$commit"
    fi
    local changed=0
    if [ "$(git -C "$tree" rev-parse HEAD)" != "$commit" ]; then
        git -C "$tree" fetch --quiet origin "$commit" || git -C "$tree" fetch --quiet origin
        git -C "$tree" checkout --quiet --detach "$commit"
        changed=1
    fi
    # the engine only needs ggml and the VST3 SDK, not the web UI's submodules
    git -C "$tree" submodule update --init engine/ggml engine/vendor/vst3sdk
    build="$tree/engine/build-linux"
    if [ "$changed" = 1 ] || [ ! -f "$build/CMakeCache.txt" ]; then
        # Vulkan stays off: upstream notes that the transcriber's Vulkan path writes wrong notes
        run cmake -S "$tree/engine" -B "$build" -G Ninja -DCMAKE_BUILD_TYPE=Release \
            -DGGML_CUDA=ON -DGGML_VULKAN=OFF -DGGML_BACKEND_DL=ON -DGGML_CPU_ALL_VARIANTS=ON \
            -DGGML_NATIVE=OFF -DGGML_CCACHE=OFF "${cuda_flags[@]}" "${rpath_flags[@]}"
    fi
    run cmake --build "$build" --target "$@" ggml-cuda $(cpu_variants "$build") -j "$jobs"
}
build_hotstep hs-midi music-midi-source.json ace-midi
build_hotstep hs-train music-train-source.json ace-train ace-caption
midi_tree="$cache/hs-midi"
train_tree="$cache/hs-train"

# ---- the studio ------------------------------------------------------------------------
cd "$root"
npm --prefix app ci --no-audit --no-fund
npm --prefix desktop ci --no-audit --no-fund
# The tauri build compiles the Rust crates and embeds the frontend; no bundle is made, the
# files are staged below. rust-toolchain.toml pins a Windows toolchain, hence RUSTUP_TOOLCHAIN.
# The cargo target lives in the cache to keep the repository clean.
export CARGO_TARGET_DIR="$cache/target"
# the config lists resources/yue2-cpp as a bundle resource, and tauri-build wants the folder to exist
resources_placeholder="$root/desktop/src-tauri/resources/yue2-cpp"
placeholder_made=0
if [ ! -d "$resources_placeholder" ]; then
    mkdir -p "$resources_placeholder"
    placeholder_made=1
fi
trap '[ "$placeholder_made" = 1 ] && rmdir "$root/desktop/src-tauri/resources/yue2-cpp" "$root/desktop/src-tauri/resources" 2>/dev/null || true' EXIT
(cd desktop && RUSTUP_TOOLCHAIN=stable run npm exec tauri build -- --no-bundle --config src-tauri/tauri.linux-release.conf.json)
studio="$CARGO_TARGET_DIR/release/YuE2-Studio"
[ -x "$studio" ] || { echo "[ERROR] the tauri build produced no executable in $CARGO_TARGET_DIR/release" >&2; exit 1; }

# ---- stage, verify, swap ---------------------------------------------------------------
mkdir -p "$(dirname "$install_dir")"
stage="$(mktemp -d "$install_dir.stage.XXXXXX")"
trap 'rm -rf "$stage"; [ "$placeholder_made" = 1 ] && rmdir "$root/desktop/src-tauri/resources/yue2-cpp" "$root/desktop/src-tauri/resources" 2>/dev/null || true' EXIT

# $1 destination folder, $2 build dir, $3 commit, $4 backend, $5... "built name:installed name"
stage_tool() {
    local dest="$1" build="$2" commit="$3" backend="$4" pair
    shift 4
    mkdir -p "$dest"
    for pair in "$@"; do
        cp "$build/${pair%%:*}" "$dest/${pair##*:}"
    done
    # -a keeps the version symlinks; the binaries find them through $ORIGIN
    find "$build" -maxdepth 1 -name 'lib*.so*' -exec cp -a {} "$dest/" \;
    printf '{"commit":"%s","backend":"%s","platform":"linux"}\n' "$commit" "$backend" > "$dest/runtime.json"
}
cp "$studio" "$stage/YuE2-Studio"
stage_tool "$stage/resources/yue2-cpp" "$yue_build" "$yue_commit" "cuda+vulkan" yue-server:yue-server
stage_tool "$stage/resources/music-midi" "$midi_tree/engine/build-linux" "$(json music-midi-source.json commit)" cuda ace-midi:music-midi
stage_tool "$stage/resources/music-train" "$train_tree/engine/build-linux" "$(json music-train-source.json commit)" cuda ace-train:music-train ace-caption:ace-caption
# the HOT-Step code is MIT, but its notice has to travel with the binaries
cp "$midi_tree/LICENSE" "$stage/resources/music-midi/LICENSE-HOT-Step.txt"
cp "$train_tree/LICENSE" "$stage/resources/music-train/LICENSE-HOT-Step.txt"
cp "$root/desktop/src-tauri/icons/128x128.png" "$stage/icon.png"

# every binary must resolve all its libraries, and the tools must at least start
fail=0
for binary in "$stage/YuE2-Studio" "$stage"/resources/*/{yue-server,music-midi,music-train,ace-caption}; do
    [ -f "$binary" ] || continue
    if ldd "$binary" | grep 'not found'; then
        echo "[ERROR] unresolved libraries for $binary" >&2
        fail=1
    fi
done
[ "$fail" = 0 ] || exit 1
# captured first: grep -q closing the pipe early would fail the pipeline under pipefail
usage="$("$stage/resources/yue2-cpp/yue-server" --help 2>&1 || true)"
echo "$usage" | grep -q 'Usage' \
    || { echo "[ERROR] yue-server --help does not print its usage" >&2; exit 1; }
for tool in music-midi/music-midi music-train/music-train music-train/ace-caption; do
    # the exit status is the tool's own business (no arguments is often an error); only the loader matters
    out="$(LD_LIBRARY_PATH="$stage/resources/${tool%/*}" timeout 20 "$stage/resources/$tool" --help 2>&1 < /dev/null || true)"
    if echo "$out" | grep -qiE 'error while loading shared libraries|cannot open shared object'; then
        echo "[ERROR] $tool does not start: $out" >&2
        exit 1
    fi
    echo "$tool starts"
done

# swap in one rename step each: the previous install stays as INSTALL_DIR.prev
if [ -e "$install_dir" ]; then
    rm -rf "$install_dir.prev"
    mv "$install_dir" "$install_dir.prev"
fi
chmod 755 "$stage"
mv "$stage" "$install_dir"

# ---- launcher and menu entry (the AppImage install keeps its own yue2-studio) ---------------
mkdir -p "$HOME/.local/bin" "$HOME/.local/share/applications"
launcher="$HOME/.local/bin/yue2-studio-native"
printf '#!/bin/sh\nexec "%s/YuE2-Studio" "$@"\n' "$install_dir" > "$launcher"
chmod 755 "$launcher"
cat > "$HOME/.local/share/applications/yue2-studio-native.desktop" <<EOF
[Desktop Entry]
Type=Application
Name=YuE2 Studio (native)
Comment=Local AI music studio, native CUDA build
Exec=$launcher
Icon=$install_dir/icon.png
Terminal=false
Categories=AudioVideo;Audio;
EOF

echo "installed into $install_dir (launcher: $launcher)"
