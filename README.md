# cosmic-comp
Compositor for the COSMIC desktop environment

> **This fork** adds ICC-profile gamut correction for wide-gamut panels, ahead of
> upstream colour management. To try it on your own account (Debian/Ubuntu/Pop!_OS):
> `scripts/try-icc.sh`, which builds, installs to `~/.local/bin`, and tells you where to
> put your profile. `scripts/try-icc.sh uninstall` undoes it. See [Using it](#using-it)
> below.

## Using it

**1. Get a profile for your panel.** It must be a *matrix/TRC* display profile, which
is what laptop vendors ship (factory calibrations such as X-Rite/Calman ones) and what
DisplayCAL or ArgyllCMS produce by default. LUT-only profiles are rejected. Sources:
- the vendor's Windows driver or colour-profile download (the `.icm` files that end up
  in `C:\Windows\System32\spool\drivers\color`),
- a review site such as notebookcheck.net, which often publishes its measured profile,
- your own measurement with DisplayCAL/ArgyllCMS and a colorimeter.

**2. Name it after the output.** Find the output name with `cosmic-randr list` (or
`ls /sys/class/drm`, dropping the `cardN-` prefix). A laptop panel is usually `eDP-1`;
external screens are names like `DP-2` or `HDMI-A-1`.

**3. Put it where the compositor looks.** It takes the first file that exists:

| Path | Applies to |
|------|------------|
| `~/.config/icc/<output>.icc` or `.icm` | that output |
| `~/.local/share/icc/<output>.icc` or `.icm` | that output |
| `~/.config/icc/profile.icc` or `.icm` | every output without its own profile |
| `~/.local/share/icc/profile.icc` or `.icm` | every output without its own profile |

`.icc` and `.icm` are the same format. Names must match exactly, so a profile already
installed under its own name (by colord or DisplayCAL, say) needs a symlink:

```sh
mkdir -p ~/.config/icc
ln -s ~/.local/share/icc/<the-profile>.icc ~/.config/icc/eDP-1.icc
```

**4. Log out and back in.** The profile is read when the output is set up, so changes
take effect on the next login. The login screen keeps the distro compositor; only your
session uses this one.

**5. Check.** Colours should change visibly, most obviously saturated reds and greens on
a wide-gamut panel. Release builds log only warnings, so silence is the good case; a
rejected or unreadable profile shows up in
`journalctl --user -b SYSLOG_IDENTIFIER=cosmic-comp | grep -i icc`.
To turn it off, move the profile away and log in again.

**If the session won't start:** switch to a text console (Ctrl+Alt+F3), log in, run
`rm ~/.local/bin/cosmic-comp`, and log in again normally.

## Development

This project uses [just](https://github.com/casey/just) as a command runner.

### Available Commands

#### Building
- `just build-debug` - Compile with debug profile
- `just build-release` - Compile with release profile (default)
- `just build-vendored` - Compile release profile with vendored dependencies
    - Requires vendoring first, which can be done with `just vendor`

#### Code Quality
- `just check` - Run clippy linter with pedantic warnings

#### Installation
- `just install` - Install binary and configuration files
- `just install-bare-session` - Install with bare session files (wayland session, systemd units)
- `just uninstall` - Remove installed files
- `just uninstall-bare-session` - Remove installed files including bare session files

#### Cleanup
- `just clean` - Run `cargo clean`
- `just clean-dist` - Run `cargo clean` and remove vendored dependencies

#### Vendoring
- `just vendor` - Vendor dependencies locally and create vendor.tar
- `just vendor-extract` - Extract vendored dependencies from vendor.tar
