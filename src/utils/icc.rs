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
    /// Effective per-channel TRC exponents, in R, G, B order. Factory panel
    /// profiles carry three distinct curves — their differences *are* the
    /// grey-balance calibration, so collapsing them to one would reintroduce
    /// the cast the profile exists to remove.
    pub gamma: [f64; 3],
}

/// Ready-to-use correction for the postprocess shader.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GamutCorrection {
    /// Linear-light sRGB → panel-native primaries, row-major.
    pub matrix: [[f32; 3]; 3],
    /// Per-channel panel TRC exponents for re-encoding after the matrix.
    pub gamma: [f32; 3],
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
    let red_gamma = match find(b"rTRC") {
        None => DEFAULT_GAMMA,
        Some(data) => trc_gamma(data, b"rTRC")?,
    };
    let channel = |sig: &[u8; 4]| -> Result<f64, IccError> {
        match find(sig) {
            None => Ok(red_gamma),
            Some(data) => trc_gamma(data, sig),
        }
    };

    Ok(IccProfile {
        red: xyz(b"rXYZ")?,
        green: xyz(b"gXYZ")?,
        blue: xyz(b"bXYZ")?,
        gamma: [red_gamma, channel(b"gTRC")?, channel(b"bTRC")?],
    })
}

/// Effective exponent of one `curv`/`para` TRC tag.
fn trc_gamma(data: &[u8], sig: &[u8; 4]) -> Result<f64, IccError> {
    if data.len() >= 12 && &data[0..4] == b"curv" {
        let count = be_u32(data, 8) as usize;
        return match count {
            0 => Ok(1.0), // identity curve per spec
            1 if data.len() >= 14 => {
                Ok(u16::from_be_bytes(data[12..14].try_into().unwrap()) as f64 / 256.0)
            }
            n if data.len() >= 12 + n * 2 => Ok(estimate_table_gamma(&data[12..12 + n * 2])),
            _ => Err(IccError::Truncated),
        };
    }
    if data.len() >= 12 && &data[0..4] == b"para" {
        return parametric_gamma(data).ok_or(IccError::Truncated);
    }
    Err(IccError::BadTagType(*sig))
}

/// Fit an exponent to a sampled TRC at its midpoint (exact for pure-gamma
/// tables, adequate for the near-gamma curves factory profiles carry).
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

/// Effective exponent of an ICC paraCurveType: evaluate the parametric
/// function at x = 0.5 and fit a pure power law through it. Exact for
/// function type 0; for piecewise types (e.g. sRGB's type 3, g = 2.4) this
/// yields the perceptually effective gamma (~2.2), which is what the shader
/// re-encode wants.
fn parametric_gamma(data: &[u8]) -> Option<f64> {
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
    let p = |i: usize| s15f16(data, 12 + i * 4);
    let x: f64 = 0.5;
    let y = match function_type {
        0 => x.powf(p(0)),
        1 | 2 => {
            let (g, a, b) = (p(0), p(1), p(2));
            let c = if function_type == 2 { p(3) } else { 0.0 };
            if a * x + b >= 0.0 {
                (a * x + b).powf(g) + c
            } else {
                c
            }
        }
        3 | 4 => {
            let (g, a, b, c, d) = (p(0), p(1), p(2), p(3), p(4));
            let (e, f) = if function_type == 4 {
                (p(5), p(6))
            } else {
                (0.0, 0.0)
            };
            if x >= d {
                (a * x + b).powf(g) + e
            } else {
                c * x + f
            }
        }
        _ => unreachable!(),
    };
    if y <= 0.0 {
        return Some(DEFAULT_GAMMA);
    }
    Some(y.ln() / x.ln())
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
    if let Some(bad) = profile.gamma.iter().find(|g| !(0.5..=5.0).contains(*g)) {
        return Err(IccError::ImplausibleGamma(*bad));
    }
    let inv_panel = invert3(panel).ok_or(IccError::Singular)?;
    let m = mul3(inv_panel, SRGB_PCS);
    Ok(GamutCorrection {
        matrix: m.map(|row| row.map(|v| v as f32)),
        gamma: profile.gamma.map(|g| g as f32),
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
        assert_close(profile.gamma[0], 2.199, 5e-3, "gamma");
    }

    #[test]
    fn estimates_gamma_from_table_trc() {
        let table: Vec<u16> = (0..256)
            .map(|i| ((i as f64 / 255.0).powf(2.2) * 65535.0).round() as u16)
            .collect();
        let bytes = synthetic_profile(SRGB_PCS_F64, TrcTag::Table(table));
        let profile = parse_icc(&bytes).expect("parses");
        assert_close(profile.gamma[0], 2.2, 0.05, "fitted gamma");
    }

    #[test]
    fn missing_trc_defaults_to_srgb_like_gamma() {
        let bytes = synthetic_profile(SRGB_PCS_F64, TrcTag::Missing);
        let profile = parse_icc(&bytes).expect("parses");
        assert_close(profile.gamma[0], 2.2, 1e-9, "default gamma");
    }

    #[test]
    fn zero_entry_trc_means_linear() {
        let bytes = synthetic_profile(SRGB_PCS_F64, TrcTag::Table(vec![]));
        let profile = parse_icc(&bytes).expect("parses");
        assert_close(profile.gamma[0], 1.0, 1e-9, "linear gamma");
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
        assert_close(profile.gamma[0], 2.4, 1e-4, "parametric gamma");
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
        for (i, g) in correction.gamma.iter().enumerate() {
            assert_close(*g as f64, 2.2, 0.1, &format!("sRGB.icc gamma[{i}]"));
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
        assert_close(profile.gamma[0], 2.2, 5e-3, "red gamma");
        assert_close(profile.gamma[1], 2.3, 5e-3, "green gamma");
        assert_close(profile.gamma[2], 2.1, 5e-3, "blue gamma");

        let correction = gamut_correction(&profile).expect("invertible");
        assert!(
            correction.gamma[0] != correction.gamma[1]
                && correction.gamma[1] != correction.gamma[2],
            "per-channel exponents must survive into the shader uniform"
        );
    }

    #[test]
    fn channel_without_trc_falls_back_to_red_not_the_default() {
        let bytes = synthetic_profile_trcs(
            SRGB_PCS_F64,
            [TrcTag::Gamma(1.8), TrcTag::Missing, TrcTag::Missing],
        );
        let profile = parse_icc(&bytes).expect("parses");
        assert_close(
            profile.gamma[1],
            profile.gamma[0],
            1e-9,
            "green follows red",
        );
        assert_close(profile.gamma[2], profile.gamma[0], 1e-9, "blue follows red");
        assert!(
            (profile.gamma[1] - DEFAULT_GAMMA).abs() > 0.1,
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

    #[test]
    fn gl_matrix_is_column_major() {
        let correction = GamutCorrection {
            matrix: [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
            gamma: [2.2, 2.2, 2.2],
        };
        assert_eq!(
            correction.gl_matrix(),
            [1.0, 4.0, 7.0, 2.0, 5.0, 8.0, 3.0, 6.0, 9.0]
        );
    }
}
