#!/usr/bin/env bash
# Build this fork's cosmic-comp (ICC gamut correction) and install it for the
# current user only, shadowing the distro compositor via ~/.local/bin.
#
#   scripts/try-icc.sh            build + install
#   scripts/try-icc.sh uninstall  restore the previous binary (or the distro one)
set -euo pipefail

repo=$(cd "$(dirname "$0")/.." && pwd)
bin_dir="$HOME/.local/bin"
dst="$bin_dir/cosmic-comp"

say() { printf '\033[1m==>\033[0m %s\n' "$*"; }
die() { printf '\033[31merror:\033[0m %s\n' "$*" >&2; exit 1; }

if [[ ${1:-} == uninstall ]]; then
    [[ -e $dst ]] || die "nothing installed at $dst"
    if [[ -e $dst.prev ]]; then
        mv -f "$dst.prev" "$dst"
        say "Restored the previous $dst."
    else
        rm -f "$dst"
        say "Removed $dst; the distro /usr/bin/cosmic-comp will be used."
    fi
    say "Log out and back in to switch compositors."
    exit 0
fi
[[ $# -eq 0 ]] || die "usage: $0 [uninstall]"

# Debian/Ubuntu/Pop!_OS build deps (debian/control). Elsewhere, install the
# equivalents yourself: cmake, mesa EGL/GBM, fontconfig, libinput, pixman,
# libseat, systemd, udev, wayland, xcb, xkbcommon, libdisplay-info headers.
if command -v dpkg-query >/dev/null; then
    deps=(cmake libegl1-mesa-dev libfontconfig-dev libgbm-dev libinput-dev
          libpixman-1-dev libseat-dev libsystemd-dev libudev-dev libwayland-dev
          libxcb1-dev libxkbcommon-dev libdisplay-info-dev pkg-config)
    missing=()
    for d in "${deps[@]}"; do
        dpkg-query -W -f='${Status}' "$d" 2>/dev/null | grep -q 'ok installed' || missing+=("$d")
    done
    if ((${#missing[@]})); then
        say "Installing build dependencies: ${missing[*]}"
        sudo apt-get install -y "${missing[@]}"
    fi
fi

# rust-toolchain.toml pins the compiler; rustup fetches it on first build.
command -v cargo >/dev/null && command -v rustup >/dev/null \
    || die "needs rustup (https://rustup.rs); a distro cargo can't honour rust-toolchain.toml"

say "Building (release; the first build takes several minutes)"
cargo build --release --manifest-path "$repo/Cargo.toml"

mkdir -p "$bin_dir"
# Keep one backup, but never back up our own build over a real previous one.
if [[ -e $dst && ! -e $dst.prev ]]; then
    cp -p "$dst" "$dst.prev"
fi
install -m0755 "$repo/target/release/cosmic-comp" "$dst.new"
mv -f "$dst.new" "$dst" # rename, so a running compositor keeps its old inode

case ":$PATH:" in
    *":$bin_dir:"*) ;;
    *) printf '\033[33mwarning:\033[0m %s is not on your PATH, so the session will keep using /usr/bin/cosmic-comp.\n' "$bin_dir" ;;
esac

outputs=$(for c in /sys/class/drm/card*-*; do
    [[ $(cat "$c/status" 2>/dev/null) == connected ]] && basename "$c" | cut -d- -f2-
done | paste -sd' ')

cat <<EOF

Installed $dst ($("$dst" --version 2>/dev/null || echo 'version unknown')).

Next:
  1. Put your panel's ICC profile at one of
       ~/.config/icc/<output>.icc   (one output; .icm also works)
       ~/.config/icc/profile.icc    (all outputs)
     ~/.local/share/icc/ is searched too, so colord/DisplayCAL installs work as-is.
     Connected outputs right now: ${outputs:-none found}
  2. Log out and back in. The login screen always uses the distro compositor;
     only your session runs this one.
  3. Colours should shift visibly. If they don't, look for a rejected profile:
       journalctl --user -b SYSLOG_IDENTIFIER=cosmic-comp | grep -i icc
     (Release builds log only warnings, so a silent journal is the good case.)

Undo: $0 uninstall, then log out and in.
If the session won't start, switch to a TTY (Ctrl+Alt+F3), log in, run
  rm ~/.local/bin/cosmic-comp
and log in again.
EOF
