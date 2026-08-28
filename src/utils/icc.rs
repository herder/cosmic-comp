// SPDX-License-Identifier: GPL-3.0-only

//! Minimal ICC display-profile support for output gamut correction.
//!
//! Parses just enough of an ICC v2/v4 matrix/TRC display profile (the kind
//! vendors like X-Rite factory-calibrate laptop panels with) to derive an
//! sRGB → panel-native compression matrix for the postprocess shader.
//!
//! ICC colorant tags (rXYZ/gXYZ/bXYZ) are stored chromatically adapted to the
//! D50 PCS. The standard sRGB colorants below are published in the same
//! D50-adapted form, so `inv(M_panel) * M_srgb` computed wholly in PCS space
//! cancels the adaptation — no Bradford transform is needed.
//!
//! LUT-based (cLUT) profiles are rejected; only matrix/TRC profiles are
//! supported, which covers factory panel profiles.

use std::path::Path;

/// sRGB colorants, D50-adapted, as found in the standard sRGB ICC profile
/// (columns: R, G, B; rows: X, Y, Z).
const SRGB_PCS: [[f64; 3]; 3] = [
    [0.4360747, 0.3850649, 0.1430804],
    [0.2225045, 0.7168786, 0.0606169],
    [0.0139322, 0.0971045, 0.7141733],
];

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Xyz {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

/// Colorimetry extracted from a matrix/TRC display profile.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IccProfile {
    /// Red colorant, D50 PCS-relative.
    pub red: Xyz,
    /// Green colorant, D50 PCS-relative.
    pub green: Xyz,
    /// Blue colorant, D50 PCS-relative.
    pub blue: Xyz,
    /// Per-channel tone response, in R, G, B order. Factory panel profiles
    /// carry three distinct curves — their differences *are* the grey-balance
    /// calibration, so collapsing them to one would reintroduce the cast the
    /// profile exists to remove.
    pub trc: [Trc; 3],
}

/// One channel's tone response, normalized to the ICC paraCurveType type-4
/// form. Every simpler curve type, and a plain gamma, is a special case:
///
/// ```text
/// Y = (a·X + b)^g + e   for X >= d
/// Y = c·X + f           for X <  d
/// ```
///
/// Collapsing this to one exponent (what a midpoint fit does) costs up to
/// **9 of 255 output codes in the deepest black** on a real sRGB-style
/// type-3 curve, where the correct code is only ~7 — so the error exceeds
/// the signal exactly where an OLED is least forgiving. Hence the full form.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Trc {
    pub g: f64,
    pub a: f64,
    pub b: f64,
    pub c: f64,
    pub d: f64,
    pub e: f64,
    pub f: f64,
}

impl Trc {
    /// A pure power law, `Y = X^g`.
    fn power(g: f64) -> Self {
        Trc {
            g,
            a: 1.0,
            b: 0.0,
            c: 0.0,
            d: 0.0,
            e: 0.0,
            f: 0.0,
        }
    }

    /// Normalize an ICC paraCurveType to the type-4 form. Semantics per
    /// ICC.1:2010 §10.16; types 0–3 differ in which parameters they omit,
    /// and notably type 2's fourth parameter is an *offset*, where type 3's
    /// is the slope of the linear segment.
    fn from_para(function_type: u16, p: &[f64]) -> Option<Self> {
        let linear_join = |a: f64, b: f64| (a != 0.0).then(|| -b / a);
        Some(match function_type {
            0 => Trc::power(p[0]),
            1 => Trc {
                g: p[0],
                a: p[1],
                b: p[2],
                c: 0.0,
                d: linear_join(p[1], p[2])?,
                e: 0.0,
                f: 0.0,
            },
            2 => Trc {
                g: p[0],
                a: p[1],
                b: p[2],
                c: 0.0,
                d: linear_join(p[1], p[2])?,
                e: p[3],
                f: p[3],
            },
            3 => Trc {
                g: p[0],
                a: p[1],
                b: p[2],
                c: p[3],
                d: p[4],
                e: 0.0,
                f: 0.0,
            },
            4 => Trc {
                g: p[0],
                a: p[1],
                b: p[2],
                c: p[3],
                d: p[4],
                e: p[5],
                f: p[6],
            },
            _ => return None,
        })
    }

    /// Decode: panel drive value → linear light.
    pub fn decode(&self, x: f64) -> f64 {
        if x >= self.d {
            let base = self.a * x + self.b;
            if base <= 0.0 {
                self.e
            } else {
                base.powf(self.g) + self.e
            }
        } else {
            self.c * x + self.f
        }
    }

    /// Linear-light value at the piecewise join — the threshold the shader
    /// branches on, since it works in linear light rather than in `X`.
    fn y_join(&self) -> f64 {
        let base = self.a * self.d + self.b;
        if base <= 0.0 {
            self.e
        } else {
            base.powf(self.g) + self.e
        }
    }

    /// Exponent of the pure power law through the curve's midpoint. Only for
    /// the plausibility check and for logging — never for the correction.
    fn effective_gamma(&self) -> f64 {
        let y = self.decode(0.5);
        if y > 0.0 {
            y.ln() / 0.5f64.ln()
        } else {
            f64::NAN
        }
    }
}

/// Ready-to-use correction for the postprocess shader.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GamutCorrection {
    /// Linear-light sRGB → panel-native primaries, row-major.
    pub matrix: [[f32; 3]; 3],
    /// Per-channel inverse-TRC coefficients for re-encoding after the matrix.
    pub trc: [TrcEncode; 3],
}

/// One channel's inverse TRC, as the shader consumes it: linear light `Y` to
/// drive value `X`, inverting [`Trc`].
///
/// ```text
/// X = ((Y - e)^(1/g) - b) / a   for Y >= y_join
/// X = (Y - f) / c               for Y <  y_join
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TrcEncode {
    pub g: f32,
    pub a: f32,
    pub b: f32,
    pub c: f32,
    pub e: f32,
    pub f: f32,
    pub y_join: f32,
}

impl TrcEncode {
    fn from_trc(t: &Trc) -> Self {
        let y_join = t.y_join();
        // Both shader branches are always evaluated, so the unused one must
        // still be finite. Where the linear segment is unreachable (y_join 0,
        // which is every pure-gamma curve) c is free, so make it a safe 1.
        let (c, y_join) = if t.c == 0.0 || y_join <= 0.0 {
            (1.0, 0.0)
        } else {
            (t.c, y_join)
        };
        TrcEncode {
            g: t.g as f32,
            a: t.a as f32,
            b: t.b as f32,
            c: c as f32,
            e: t.e as f32,
            f: t.f as f32,
            y_join: y_join as f32,
        }
    }

    /// Encode: linear light → drive value. Mirrors the shader exactly, so a
    /// test can hold the two to the same numbers.
    pub fn encode(&self, y: f32) -> f32 {
        if y >= self.y_join {
            ((y - self.e).max(0.0).powf(1.0 / self.g) - self.b) / self.a
        } else {
            (y - self.f) / self.c
        }
    }
}

impl GamutCorrection {
    /// Matrix in column-major order, as GLES2 `uniformMatrix3fv` expects
    /// (ES 2.0 requires transpose=false, so we transpose CPU-side).
    pub fn gl_matrix(&self) -> [f32; 9] {
        let m = &self.matrix;
        [
            m[0][0], m[1][0], m[2][0], //
            m[0][1], m[1][1], m[2][1], //
            m[0][2], m[1][2], m[2][2],
        ]
    }

    /// How far outside the panel's gamut the sRGB primaries fall, if at all.
    ///
    /// A negative matrix coefficient means an sRGB primary is not reproducible
    /// on this panel, so the shader's clamp will hard-clip it and shift that
    /// hue. Wide-gamut panels contain sRGB and return `None`; panels that do
    /// not fully cover it do not, and the caller should say so out loud.
    pub fn srgb_coverage_deficit(&self) -> Option<f32> {
        // Published colorant sets are rounded, so a coefficient that is truly
        // zero lands a hair either side of it. Below a quarter of an 8-bit
        // code there is nothing to clip and nothing to report.
        const NOISE_FLOOR: f32 = 1e-3;
        self.matrix
            .iter()
            .flatten()
            .copied()
            .filter(|v| *v < -NOISE_FLOOR)
            .fold(None, |worst: Option<f32>, v| {
                Some(worst.map_or(-v, |w| w.max(-v)))
            })
    }

    /// Shader uniform name/value pairs for the inverse TRC, one `vec3` per
    /// coefficient across the R, G, B channels. Names match offscreen.frag.
    pub fn trc_uniforms(&self) -> [(&'static str, (f32, f32, f32)); 7] {
        let [r, g, b] = &self.trc;
        [
            ("trc_g", (r.g, g.g, b.g)),
            ("trc_a", (r.a, g.a, b.a)),
            ("trc_b", (r.b, g.b, b.b)),
            ("trc_c", (r.c, g.c, b.c)),
            ("trc_e", (r.e, g.e, b.e)),
            ("trc_f", (r.f, g.f, b.f)),
            ("trc_y_join", (r.y_join, g.y_join, b.y_join)),
        ]
    }
}

#[derive(Debug, thiserror::Error)]
pub enum IccError {
    #[error("buffer too small for ICC header")]
    TooShort,
    #[error("missing 'acsp' profile signature")]
    BadMagic,
    #[error("tag table out of bounds")]
    Truncated,
    #[error("missing required tag {0:?} (LUT-based profiles are unsupported)")]
    MissingTag([u8; 4]),
    #[error("unsupported tag type for {0:?}")]
    BadTagType([u8; 4]),
    #[error("colorant matrix is singular")]
    Singular,
    #[error("TRC gamma {0} is outside the plausible display range")]
    ImplausibleGamma(f64),
    #[error("not an RGB/XYZ display profile (class {class:?}, space {space:?}, PCS {pcs:?})")]
    WrongProfileKind {
        class: [u8; 4],
        space: [u8; 4],
        pcs: [u8; 4],
    },
}

/// TRC exponent assumed when a profile carries no (usable) TRC tag.
const DEFAULT_GAMMA: f64 = 2.2;

fn be_u32(bytes: &[u8], offset: usize) -> u32 {
    u32::from_be_bytes(bytes[offset..offset + 4].try_into().unwrap())
}

/// ICC s15Fixed16Number.
fn s15f16(bytes: &[u8], offset: usize) -> f64 {
    i32::from_be_bytes(bytes[offset..offset + 4].try_into().unwrap()) as f64 / 65536.0
}

/// Parse the colorant + TRC tags out of an ICC display profile.
pub fn parse_icc(bytes: &[u8]) -> Result<IccProfile, IccError> {
    if bytes.len() < 132 {
        return Err(IccError::TooShort);
    }
    if &bytes[36..40] != b"acsp" {
        return Err(IccError::BadMagic);
    }
    // Header fields per ICC.1:2010 §7.2: device class at 12, data colour
    // space at 16, PCS at 20. A printer profile or a Lab-PCS profile can
    // carry tags of the right shape and would otherwise parse into nonsense
    // that only shows up as wrong colour on screen.
    let (class, space, pcs) = (&bytes[12..16], &bytes[16..20], &bytes[20..24]);
    if class != b"mntr" || space != b"RGB " || pcs != b"XYZ " {
        return Err(IccError::WrongProfileKind {
            class: class.try_into().unwrap(),
            space: space.try_into().unwrap(),
            pcs: pcs.try_into().unwrap(),
        });
    }

    let tag_count = be_u32(bytes, 128) as usize;
    let table_end = 132usize
        .checked_add(tag_count.checked_mul(12).ok_or(IccError::Truncated)?)
        .ok_or(IccError::Truncated)?;
    if bytes.len() < table_end {
        return Err(IccError::Truncated);
    }

    // Validate every entry up front so a truncated file fails loudly even if
    // we never read the clipped tag.
    let mut tags = Vec::with_capacity(tag_count);
    for i in 0..tag_count {
        let entry = 132 + i * 12;
        let sig: [u8; 4] = bytes[entry..entry + 4].try_into().unwrap();
        let offset = be_u32(bytes, entry + 4) as usize;
        let size = be_u32(bytes, entry + 8) as usize;
        let end = offset.checked_add(size).ok_or(IccError::Truncated)?;
        if end > bytes.len() {
            return Err(IccError::Truncated);
        }
        tags.push((sig, &bytes[offset..end]));
    }
    let find = |sig: &[u8; 4]| tags.iter().find(|(s, _)| s == sig).map(|(_, d)| *d);

    let xyz = |sig: &[u8; 4]| -> Result<Xyz, IccError> {
        let data = find(sig).ok_or(IccError::MissingTag(*sig))?;
        if data.len() < 20 || &data[0..4] != b"XYZ " {
            return Err(IccError::BadTagType(*sig));
        }
        Ok(Xyz {
            x: s15f16(data, 8),
            y: s15f16(data, 12),
            z: s15f16(data, 16),
        })
    };

    // Red is the reference: a profile carrying only rTRC means "same for all",
    // which is a better fallback for the other two than DEFAULT_GAMMA.
    let red = match find(b"rTRC") {
        None => Trc::power(DEFAULT_GAMMA),
        Some(data) => trc_curve(data, b"rTRC")?,
    };
    let channel = |sig: &[u8; 4]| -> Result<Trc, IccError> {
        match find(sig) {
            None => Ok(red),
            Some(data) => trc_curve(data, sig),
        }
    };

    Ok(IccProfile {
        red: xyz(b"rXYZ")?,
        green: xyz(b"gXYZ")?,
        blue: xyz(b"bXYZ")?,
        trc: [red, channel(b"gTRC")?, channel(b"bTRC")?],
    })
}

/// One `curv`/`para` TRC tag, normalized to [`Trc`].
fn trc_curve(data: &[u8], sig: &[u8; 4]) -> Result<Trc, IccError> {
    if data.len() >= 12 && &data[0..4] == b"curv" {
        let count = be_u32(data, 8) as usize;
        return match count {
            0 => Ok(Trc::power(1.0)), // identity curve per spec
            1 if data.len() >= 14 => Ok(Trc::power(
                u16::from_be_bytes(data[12..14].try_into().unwrap()) as f64 / 256.0,
            )),
            // A sampled table has no closed form; a fitted power law is the
            // best we can do without uploading the whole curve to the GPU.
            n if data.len() >= 12 + n * 2 => {
                Ok(Trc::power(estimate_table_gamma(&data[12..12 + n * 2])))
            }
            _ => Err(IccError::Truncated),
        };
    }
    if data.len() >= 12 && &data[0..4] == b"para" {
        return parametric_trc(data).ok_or(IccError::Truncated);
    }
    Err(IccError::BadTagType(*sig))
}

/// Fit an exponent to a sampled TRC at its midpoint.
///
/// Exact for a pure-gamma table, which is the case a fit is actually for.
/// Least squares on log(y) = g·log(x) looks like the better answer and is
/// not: it minimizes *relative* error, so a curve with a linear toe (an
/// sRGB-shaped table) drags the fit down badly — 16.5 output codes of peak
/// error against the midpoint fit's 1.3, and still worse when restricted to
/// the upper range. Measured, not assumed; see the accompanying test.
fn estimate_table_gamma(samples: &[u8]) -> f64 {
    let n = samples.len() / 2;
    let at = |i: usize| u16::from_be_bytes(samples[i * 2..i * 2 + 2].try_into().unwrap());
    let mid = n / 2;
    let x = mid as f64 / (n - 1) as f64;
    let y = at(mid) as f64 / 65535.0;
    if x <= 0.0 || x >= 1.0 || y <= 0.0 {
        return DEFAULT_GAMMA;
    }
    y.ln() / x.ln()
}

/// Parse an ICC paraCurveType into the normalized [`Trc`] form.
fn parametric_trc(data: &[u8]) -> Option<Trc> {
    let function_type = u16::from_be_bytes(data[8..10].try_into().unwrap());
    let param_count = match function_type {
        0 => 1,
        1 => 3,
        2 => 4,
        3 => 5,
        4 => 7,
        _ => return None,
    };
    if data.len() < 12 + param_count * 4 {
        return None;
    }
    let p: Vec<f64> = (0..param_count).map(|i| s15f16(data, 12 + i * 4)).collect();
    Trc::from_para(function_type, &p)
}

fn invert3(m: [[f64; 3]; 3]) -> Option<[[f64; 3]; 3]> {
    let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    if det.abs() < 1e-9 {
        return None;
    }
    let inv_det = 1.0 / det;
    let mut inv = [[0.0; 3]; 3];
    for row in 0..3 {
        for col in 0..3 {
            // Cofactor expansion; note the (col, row) transpose for the adjugate.
            let a = m[(row + 1) % 3][(col + 1) % 3];
            let b = m[(row + 2) % 3][(col + 2) % 3];
            let c = m[(row + 1) % 3][(col + 2) % 3];
            let d = m[(row + 2) % 3][(col + 1) % 3];
            inv[col][row] = (a * b - c * d) * inv_det;
        }
    }
    Some(inv)
}

fn mul3(a: [[f64; 3]; 3], b: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut out = [[0.0; 3]; 3];
    for row in 0..3 {
        for col in 0..3 {
            out[row][col] = (0..3).map(|k| a[row][k] * b[k][col]).sum();
        }
    }
    out
}

/// Derive the sRGB → panel correction from parsed colorimetry.
///
/// Both colorant matrices are D50 PCS-relative, so the chromatic adaptation
/// cancels and the product is the D65-native transform.
pub fn gamut_correction(profile: &IccProfile) -> Result<GamutCorrection, IccError> {
    let panel = [
        [profile.red.x, profile.green.x, profile.blue.x],
        [profile.red.y, profile.green.y, profile.blue.y],
        [profile.red.z, profile.green.z, profile.blue.z],
    ];
    for trc in &profile.trc {
        let effective = trc.effective_gamma();
        if !(0.5..=5.0).contains(&effective) || trc.a == 0.0 {
            return Err(IccError::ImplausibleGamma(effective));
        }
    }
    let inv_panel = invert3(panel).ok_or(IccError::Singular)?;
    let m = mul3(inv_panel, SRGB_PCS);
    Ok(GamutCorrection {
        matrix: m.map(|row| row.map(|v| v as f32)),
        trc: [
            TrcEncode::from_trc(&profile.trc[0]),
            TrcEncode::from_trc(&profile.trc[1]),
            TrcEncode::from_trc(&profile.trc[2]),
        ],
    })
}

/// Candidate profile paths for an output, most specific first.
///
/// `~/.local/share/icc` is where colord, DisplayCAL and argyll actually install
/// profiles, so it is searched alongside our own `~/.config/icc` override dir.
/// An output-specific profile in either dir beats a generic one in either.
fn profile_candidates(
    config_dir: Option<&Path>,
    data_dir: Option<&Path>,
    output_name: &str,
) -> Vec<std::path::PathBuf> {
    let dirs: Vec<_> = [config_dir, data_dir]
        .into_iter()
        .flatten()
        .map(|d| d.join("icc"))
        .collect();

    let mut specific = Vec::new();
    let mut generic = Vec::new();
    for dir in &dirs {
        // Both extensions denote the same format; tools disagree on which to write.
        for ext in ["icc", "icm"] {
            specific.push(dir.join(format!("{output_name}.{ext}")));
            generic.push(dir.join(format!("profile.{ext}")));
        }
    }
    // An output-specific profile in any dir beats a generic one anywhere.
    specific.extend(generic);
    specific
}

/// Load a correction for an output. Searched paths are those of
/// [`profile_candidates`]; any error is logged and treated as "no profile".
pub fn load_for_output(output_name: &str) -> Option<GamutCorrection> {
    let home = std::env::var_os("HOME").map(std::path::PathBuf::from);
    let config_dir = std::env::var_os("XDG_CONFIG_HOME")
        .map(std::path::PathBuf::from)
        .or_else(|| home.as_ref().map(|h| h.join(".config")));
    let data_dir = std::env::var_os("XDG_DATA_HOME")
        .map(std::path::PathBuf::from)
        .or_else(|| home.as_ref().map(|h| h.join(".local/share")));

    profile_candidates(config_dir.as_deref(), data_dir.as_deref(), output_name)
        .into_iter()
        .find(|p| p.exists())
        .and_then(|path| match std::fs::read(&path) {
            Ok(bytes) => match parse_icc(&bytes).and_then(|p| gamut_correction(&p)) {
                Ok(correction) => {
                    if let Some(deficit) = correction.srgb_coverage_deficit() {
                        tracing::warn!(
                            ?path,
                            deficit,
                            "Panel gamut does not contain sRGB; out-of-gamut colors \
                             will be hard-clipped and those hues will shift"
                        );
                    }
                    tracing::info!(?path, ?correction, "Loaded ICC gamut correction");
                    Some(correction)
                }
                Err(err) => {
                    tracing::warn!(?path, ?err, "Ignoring unusable ICC profile");
                    None
                }
            },
            Err(err) => {
                tracing::warn!(?path, ?err, "Failed to read ICC profile");
                None
            }
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s15f16(v: f64) -> [u8; 4] {
        (((v * 65536.0).round()) as i32).to_be_bytes()
    }

    fn xyz_tag(col: usize, colorants: &[[f64; 3]; 3]) -> Vec<u8> {
        let mut data = b"XYZ \0\0\0\0".to_vec();
        for row in 0..3 {
            data.extend_from_slice(&s15f16(colorants[row][col]));
        }
        data
    }

    fn trc_bytes(trc: &TrcTag) -> Vec<u8> {
        match trc {
            TrcTag::Gamma(g) => {
                let mut d = b"curv\0\0\0\0".to_vec();
                d.extend_from_slice(&1u32.to_be_bytes());
                d.extend_from_slice(&(((g * 256.0).round()) as u16).to_be_bytes());
                d
            }
            TrcTag::Table(samples) => {
                let mut d = b"curv\0\0\0\0".to_vec();
                d.extend_from_slice(&(samples.len() as u32).to_be_bytes());
                for s in samples {
                    d.extend_from_slice(&s.to_be_bytes());
                }
                d
            }
            TrcTag::Missing => Vec::new(),
            TrcTag::Para(g) => {
                let mut d = b"para\0\0\0\0".to_vec();
                d.extend_from_slice(&0u16.to_be_bytes()); // function type 0: pure gamma
                d.extend_from_slice(&0u16.to_be_bytes()); // reserved
                d.extend_from_slice(&(((g * 65536.0).round()) as i32).to_be_bytes());
                d
            }
        }
    }

    /// Build a minimal matrix/TRC ICC display profile.
    /// `colorants` are (r, g, b) columns as XYZ triplets, D50 PCS-relative.
    /// `trcs` become the rTRC/gTRC/bTRC tags; `Missing` omits that tag.
    fn synthetic_profile_trcs(colorants: [[f64; 3]; 3], trcs: [TrcTag; 3]) -> Vec<u8> {
        let mut tags: Vec<([u8; 4], Vec<u8>)> = vec![
            (*b"rXYZ", xyz_tag(0, &colorants)),
            (*b"gXYZ", xyz_tag(1, &colorants)),
            (*b"bXYZ", xyz_tag(2, &colorants)),
        ];
        for (sig, trc) in [b"rTRC", b"gTRC", b"bTRC"].into_iter().zip(&trcs) {
            let data = trc_bytes(trc);
            if !data.is_empty() {
                tags.push((*sig, data));
            }
        }

        let tag_table_len = 4 + tags.len() * 12;
        let mut offset = 128 + tag_table_len;
        let mut header = vec![0u8; 128];
        header[12..16].copy_from_slice(b"mntr");
        header[16..20].copy_from_slice(b"RGB ");
        header[20..24].copy_from_slice(b"XYZ ");
        header[36..40].copy_from_slice(b"acsp");
        let mut table = (tags.len() as u32).to_be_bytes().to_vec();
        let mut body = Vec::new();
        for (sig, data) in &tags {
            table.extend_from_slice(sig);
            table.extend_from_slice(&(offset as u32).to_be_bytes());
            table.extend_from_slice(&(data.len() as u32).to_be_bytes());
            offset += data.len();
            body.extend_from_slice(data);
        }
        let mut out = header;
        out.extend_from_slice(&table);
        out.extend_from_slice(&body);
        let total = (out.len() as u32).to_be_bytes();
        out[0..4].copy_from_slice(&total);
        out
    }

    /// The common case: one curve mirrored across all three channels.
    fn synthetic_profile(colorants: [[f64; 3]; 3], trc: TrcTag) -> Vec<u8> {
        synthetic_profile_trcs(colorants, [trc.clone(), trc.clone(), trc])
    }

    #[derive(Clone)]
    enum TrcTag {
        Gamma(f64),
        Table(Vec<u16>),
        Para(f64),
        Missing,
    }

    /// Display P3 colorants as stored (D50-adapted) in Apple's Display P3 ICC.
    const P3_PCS: [[f64; 3]; 3] = [
        [0.51512, 0.29198, 0.15710],
        [0.24120, 0.69225, 0.06657],
        [-0.00105, 0.04189, 0.78407],
    ];

    /// Reference linear sRGB → Display P3 matrix (both D65; adaptation cancels).
    const SRGB_TO_P3: [[f64; 3]; 3] = [
        [0.8224886, 0.1775114, 0.0],
        [0.0331941, 0.9668059, 0.0],
        [0.0170827, 0.0723974, 0.9105199],
    ];

    fn assert_close(actual: f64, expected: f64, tol: f64, what: &str) {
        assert!(
            (actual - expected).abs() < tol,
            "{what}: expected {expected}, got {actual} (tol {tol})"
        );
    }

    #[test]
    fn rejects_short_buffer() {
        assert!(matches!(parse_icc(&[0u8; 10]), Err(IccError::TooShort)));
    }

    #[test]
    fn rejects_bad_magic() {
        let mut bytes = synthetic_profile(SRGB_PCS_F64, TrcTag::Gamma(2.2));
        bytes[36..40].copy_from_slice(b"nope");
        assert!(matches!(parse_icc(&bytes), Err(IccError::BadMagic)));
    }

    #[test]
    fn rejects_profile_without_colorants() {
        // Header + zero tags: the shape of a cLUT-only profile for our purposes.
        let mut bytes = vec![0u8; 132];
        bytes[12..16].copy_from_slice(b"mntr");
        bytes[16..20].copy_from_slice(b"RGB ");
        bytes[20..24].copy_from_slice(b"XYZ ");
        bytes[36..40].copy_from_slice(b"acsp");
        bytes[0..4].copy_from_slice(&132u32.to_be_bytes());
        // tag count = 0 at 128..132 already zeroed
        assert!(matches!(parse_icc(&bytes), Err(IccError::MissingTag(_))));
    }

    #[test]
    fn rejects_truncated_tag_data() {
        let mut bytes = synthetic_profile(SRGB_PCS_F64, TrcTag::Gamma(2.2));
        bytes.truncate(bytes.len() - 4);
        assert!(matches!(parse_icc(&bytes), Err(IccError::Truncated)));
    }

    const SRGB_PCS_F64: [[f64; 3]; 3] = SRGB_PCS;

    #[test]
    fn parses_colorants_from_synthetic_srgb_profile() {
        let bytes = synthetic_profile(SRGB_PCS_F64, TrcTag::Gamma(2.2));
        let profile = parse_icc(&bytes).expect("parses");
        assert_close(profile.red.x, 0.4360747, 1e-4, "red.x");
        assert_close(profile.red.y, 0.2225045, 1e-4, "red.y");
        assert_close(profile.green.y, 0.7168786, 1e-4, "green.y");
        assert_close(profile.blue.z, 0.7141733, 1e-4, "blue.z");
    }

    #[test]
    fn parses_single_entry_gamma_trc() {
        let bytes = synthetic_profile(SRGB_PCS_F64, TrcTag::Gamma(2.2));
        let profile = parse_icc(&bytes).expect("parses");
        // u8.8 quantization: 2.2 * 256 rounds to 563 -> 2.19921875
        assert_close(profile.trc[0].effective_gamma(), 2.199, 5e-3, "gamma");
    }

    #[test]
    fn estimates_gamma_from_table_trc() {
        let table: Vec<u16> = (0..256)
            .map(|i| ((i as f64 / 255.0).powf(2.2) * 65535.0).round() as u16)
            .collect();
        let bytes = synthetic_profile(SRGB_PCS_F64, TrcTag::Table(table));
        let profile = parse_icc(&bytes).expect("parses");
        assert_close(profile.trc[0].effective_gamma(), 2.2, 0.05, "fitted gamma");
    }

    #[test]
    fn missing_trc_defaults_to_srgb_like_gamma() {
        let bytes = synthetic_profile(SRGB_PCS_F64, TrcTag::Missing);
        let profile = parse_icc(&bytes).expect("parses");
        assert_close(profile.trc[0].effective_gamma(), 2.2, 1e-9, "default gamma");
    }

    #[test]
    fn zero_entry_trc_means_linear() {
        let bytes = synthetic_profile(SRGB_PCS_F64, TrcTag::Table(vec![]));
        let profile = parse_icc(&bytes).expect("parses");
        assert_close(profile.trc[0].effective_gamma(), 1.0, 1e-9, "linear gamma");
    }

    #[test]
    fn srgb_panel_yields_identity_correction() {
        let bytes = synthetic_profile(SRGB_PCS_F64, TrcTag::Gamma(2.2));
        let profile = parse_icc(&bytes).expect("parses");
        let correction = gamut_correction(&profile).expect("invertible");
        for row in 0..3 {
            for col in 0..3 {
                let expected = if row == col { 1.0 } else { 0.0 };
                assert_close(
                    correction.matrix[row][col] as f64,
                    expected,
                    1e-3,
                    &format!("matrix[{row}][{col}]"),
                );
            }
        }
    }

    #[test]
    fn p3_panel_yields_reference_srgb_to_p3_matrix() {
        let bytes = synthetic_profile(P3_PCS, TrcTag::Gamma(2.2));
        let profile = parse_icc(&bytes).expect("parses");
        let correction = gamut_correction(&profile).expect("invertible");
        for row in 0..3 {
            for col in 0..3 {
                assert_close(
                    correction.matrix[row][col] as f64,
                    SRGB_TO_P3[row][col],
                    5e-3,
                    &format!("matrix[{row}][{col}]"),
                );
            }
        }
    }

    #[test]
    fn degenerate_colorants_are_rejected() {
        // All three colorants identical -> singular matrix.
        let bad = [[0.3, 0.3, 0.3], [0.3, 0.3, 0.3], [0.3, 0.3, 0.3]];
        let bytes = synthetic_profile(bad, TrcTag::Gamma(2.2));
        let profile = parse_icc(&bytes).expect("parses");
        assert!(matches!(
            gamut_correction(&profile),
            Err(IccError::Singular)
        ));
    }

    #[test]
    fn implausible_gamma_is_rejected() {
        // A malformed TRC declaring gamma ~0 must not reach the shader,
        // where 1.0/gamma would divide by zero.
        let bytes = synthetic_profile(SRGB_PCS_F64, TrcTag::Gamma(0.0));
        let profile = parse_icc(&bytes).expect("parses");
        assert!(matches!(
            gamut_correction(&profile),
            Err(IccError::ImplausibleGamma(_))
        ));
    }

    #[test]
    fn parses_parametric_gamma_trc() {
        // ICC v4 paraCurveType (as Apple profiles use); first param is the exponent.
        let bytes = synthetic_profile(SRGB_PCS_F64, TrcTag::Para(2.4));
        let profile = parse_icc(&bytes).expect("parses");
        assert_close(
            profile.trc[0].effective_gamma(),
            2.4,
            1e-4,
            "parametric gamma",
        );
    }

    #[test]
    fn parses_real_colord_srgb_profile_if_present() {
        // Integration smoke test against a real-world file; skipped silently
        // where colord's profiles aren't installed.
        let Ok(bytes) = std::fs::read("/usr/share/color/icc/colord/sRGB.icc") else {
            return;
        };
        let profile = parse_icc(&bytes).expect("real sRGB.icc parses");
        let correction = gamut_correction(&profile).expect("invertible");
        for row in 0..3 {
            for col in 0..3 {
                let expected = if row == col { 1.0 } else { 0.0 };
                assert_close(
                    correction.matrix[row][col] as f64,
                    expected,
                    2e-2,
                    &format!("sRGB.icc ~identity [{row}][{col}]"),
                );
            }
        }
        // sRGB's piecewise EOTF fits a pure exponent of ~2.2 at midpoint.
        for (i, t) in correction.trc.iter().enumerate() {
            assert_close(t.g as f64, 2.4, 0.1, &format!("sRGB.icc exponent[{i}]"));
            assert!(t.y_join > 0.0, "sRGB.icc keeps its linear toe [{i}]");
        }
    }

    #[test]
    fn per_channel_trc_curves_are_kept_distinct() {
        // The grey-balance calibration in a factory profile *is* the difference
        // between the three curves; collapsing them reintroduces the cast.
        let bytes = synthetic_profile_trcs(
            SRGB_PCS_F64,
            [TrcTag::Gamma(2.2), TrcTag::Gamma(2.3), TrcTag::Gamma(2.1)],
        );
        let profile = parse_icc(&bytes).expect("parses");
        assert_close(profile.trc[0].effective_gamma(), 2.2, 5e-3, "red gamma");
        assert_close(profile.trc[1].effective_gamma(), 2.3, 5e-3, "green gamma");
        assert_close(profile.trc[2].effective_gamma(), 2.1, 5e-3, "blue gamma");

        let correction = gamut_correction(&profile).expect("invertible");
        assert!(
            correction.trc[0].g != correction.trc[1].g
                && correction.trc[1].g != correction.trc[2].g,
            "per-channel curves must survive into the shader uniforms"
        );
    }

    #[test]
    fn channel_without_trc_falls_back_to_red_not_the_default() {
        let bytes = synthetic_profile_trcs(
            SRGB_PCS_F64,
            [TrcTag::Gamma(1.8), TrcTag::Missing, TrcTag::Missing],
        );
        let profile = parse_icc(&bytes).expect("parses");
        assert_eq!(profile.trc[1], profile.trc[0], "green follows red");
        assert_eq!(profile.trc[2], profile.trc[0], "blue follows red");
        assert!(
            (profile.trc[1].effective_gamma() - DEFAULT_GAMMA).abs() > 0.1,
            "an rTRC-only profile means 'same for all', not 'assume 2.2'"
        );
    }

    #[test]
    fn implausible_gamma_in_any_channel_is_rejected() {
        // Validation must cover all three, not just the one it used to read.
        let bytes = synthetic_profile_trcs(
            SRGB_PCS_F64,
            [TrcTag::Gamma(2.2), TrcTag::Gamma(2.2), TrcTag::Gamma(0.0)],
        );
        let profile = parse_icc(&bytes).expect("parses");
        assert!(matches!(
            gamut_correction(&profile),
            Err(IccError::ImplausibleGamma(_))
        ));
    }

    #[test]
    fn profile_search_prefers_output_specific_and_covers_the_standard_data_dir() {
        let config = std::path::Path::new("/cfg");
        let data = std::path::Path::new("/data");
        let paths = profile_candidates(Some(config), Some(data), "eDP-1");

        let first_generic = paths
            .iter()
            .position(|p| p.ends_with("profile.icc"))
            .expect("generic candidate present");
        let last_specific = paths
            .iter()
            .rposition(|p| p.to_string_lossy().contains("eDP-1"))
            .expect("output-specific candidate present");
        assert!(
            last_specific < first_generic,
            "an output-specific profile must outrank a generic one: {paths:?}"
        );

        // colord, DisplayCAL and argyll install here; the old code never looked.
        assert!(
            paths.contains(&data.join("icc").join("eDP-1.icc")),
            "the standard data dir must be searched: {paths:?}"
        );
        assert!(
            paths.contains(&config.join("icc").join("eDP-1.icm")),
            "the config override dir must still be searched: {paths:?}"
        );
    }

    /// The sRGB TRC as a paraCurveType type 3, as real profiles store it.
    const SRGB_PARA: [f64; 5] = [2.39999, 0.94786, 0.05214, 0.07739, 0.04045];

    #[test]
    fn shader_encode_inverts_the_parsed_curve_everywhere() {
        // The reason the piecewise form is carried through at all: the encode
        // the shader performs must invert the profile's decode across the
        // whole range, the deep shadows included.
        let trc = Trc::from_para(3, &SRGB_PARA).expect("type 3 parses");
        let encode = TrcEncode::from_trc(&trc);
        for i in 0..=1000 {
            let x = i as f64 / 1000.0;
            let round_tripped = encode.encode(trc.decode(x) as f32) as f64;
            assert!(
                (round_tripped - x).abs() < 1e-3,
                "x={x} decoded to {} and came back {round_tripped}",
                trc.decode(x)
            );
        }
    }

    #[test]
    fn collapsing_the_curve_to_one_exponent_would_wreck_the_shadows() {
        // Documents the cost this complexity buys off: a single fitted
        // exponent is worth ~9 of 255 output codes in the deepest black,
        // where the correct code is only about 7.
        let trc = Trc::from_para(3, &SRGB_PARA).expect("type 3 parses");
        let exact = TrcEncode::from_trc(&trc);
        let collapsed = TrcEncode::from_trc(&Trc::power(trc.effective_gamma()));

        let (worst_codes, at) = (1..=4095)
            .map(|i| {
                let y = i as f32 / 4095.0;
                ((collapsed.encode(y) - exact.encode(y)).abs() * 255.0, y)
            })
            .fold((0.0f32, 0.0f32), |a, b| if b.0 > a.0 { b } else { a });

        assert!(
            worst_codes > 5.0,
            "expected the collapse to be visibly wrong, got {worst_codes:.2} codes"
        );
        assert!(
            at < 0.05,
            "and wrong in the shadows specifically, got worst at linear={at}"
        );
    }

    #[test]
    fn rejects_a_profile_that_is_not_an_rgb_xyz_display_profile() {
        // A CMYK printer profile can carry tags of the right shape; parsing it
        // as a panel profile would silently produce nonsense colour.
        let mut bytes = synthetic_profile(SRGB_PCS_F64, TrcTag::Gamma(2.2));
        bytes[12..16].copy_from_slice(b"prtr");
        assert!(matches!(
            parse_icc(&bytes),
            Err(IccError::WrongProfileKind { .. })
        ));

        let mut bytes = synthetic_profile(SRGB_PCS_F64, TrcTag::Gamma(2.2));
        bytes[20..24].copy_from_slice(b"Lab ");
        assert!(matches!(
            parse_icc(&bytes),
            Err(IccError::WrongProfileKind { .. })
        ));
    }

    #[test]
    fn a_panel_narrower_than_srgb_is_reported_not_silently_clipped() {
        // sRGB primaries pulled 30% toward white — a panel that cannot
        // reproduce them. The shader clamp would hard-clip and shift the hue,
        // so the deficit has to reach the log.
        const NARROW_PCS: [[f64; 3]; 3] = [
            [0.4016743, 0.3659674, 0.1965783],
            [0.2557531, 0.6018150, 0.1424318],
            [0.0922735, 0.1504941, 0.5824423],
        ];
        let bytes = synthetic_profile(NARROW_PCS, TrcTag::Gamma(2.2));
        let profile = parse_icc(&bytes).expect("parses");
        let correction = gamut_correction(&profile).expect("invertible");
        let deficit = correction
            .srgb_coverage_deficit()
            .expect("a narrow panel must report a deficit");
        assert_close(deficit as f64, 0.14286, 1e-3, "worst deficit");

        // A wide-gamut panel contains sRGB and must stay quiet.
        let p3 = parse_icc(&synthetic_profile(P3_PCS, TrcTag::Gamma(2.2))).expect("parses");
        assert!(
            gamut_correction(&p3)
                .expect("invertible")
                .srgb_coverage_deficit()
                .is_none(),
            "P3 contains sRGB and must not warn"
        );
    }

    #[test]
    fn table_trc_fit_keeps_the_midpoint_rather_than_least_squares() {
        // Guards against a plausible-looking "improvement". Least squares on
        // log(y) = g·log(x) minimizes relative error, so an sRGB-shaped table
        // drags it toward the toe and it loses badly to one midpoint sample.
        let trc = Trc::from_para(3, &SRGB_PARA).expect("type 3 parses");
        let n = 1024usize;
        let samples: Vec<u16> = (0..n)
            .map(|i| (trc.decode(i as f64 / (n - 1) as f64) * 65535.0).round() as u16)
            .collect();
        let value = |i: usize| samples[i] as f64 / 65535.0;
        let peak_err = |g: f64| {
            (1..n)
                .map(|i| (i as f64 / (n - 1) as f64).powf(g) - value(i))
                .fold(0.0f64, |m, e| m.max(e.abs()))
        };

        let fitted = parse_icc(&synthetic_profile(
            SRGB_PCS_F64,
            TrcTag::Table(samples.clone()),
        ))
        .expect("parses")
        .trc[0]
            .g;
        let least_squares = {
            let (mut num, mut den) = (0.0, 0.0);
            for i in 1..n - 1 {
                let (x, y) = ((i as f64 / (n - 1) as f64).ln(), value(i).ln());
                num += x * y;
                den += x * x;
            }
            num / den
        };
        assert!(
            peak_err(fitted) < peak_err(least_squares),
            "midpoint fit {fitted} (err {:.5}) must stay ahead of least squares \
             {least_squares} (err {:.5})",
            peak_err(fitted),
            peak_err(least_squares)
        );

        // And a table that really is a power law is recovered exactly.
        let pure: Vec<u16> = (0..n)
            .map(|i| ((i as f64 / (n - 1) as f64).powf(2.2) * 65535.0).round() as u16)
            .collect();
        let g = parse_icc(&synthetic_profile(SRGB_PCS_F64, TrcTag::Table(pure)))
            .expect("parses")
            .trc[0]
            .g;
        assert_close(g, 2.2, 1e-4, "pure power table");
    }

    #[test]
    fn gl_matrix_is_column_major() {
        let correction = GamutCorrection {
            matrix: [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
            trc: [TrcEncode::from_trc(&Trc::power(2.2)); 3],
        };
        assert_eq!(
            correction.gl_matrix(),
            [1.0, 4.0, 7.0, 2.0, 5.0, 8.0, 3.0, 6.0, 9.0]
        );
    }
}
